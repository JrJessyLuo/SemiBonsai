from uncertainty_detection import *

import argparse
import glob
import json
import os
import pickle
import time
from typing import Any, Dict, List
import yaml
from loguru import logger
from tqdm import tqdm
from pruning import prune_phrase_groundings_with_layered_tree

from utils.basic_utils import (
    load_table_meta_from_layered_tree,
    build_union_meta_for_raw,
    extract_json_from_text,
    read_jsonl
)
from utils.api_utils import llm_generate_setup as llm_generate
from utils.constants import MODEL_MAP


def load_processed_ids(out_dir: str) -> set:
    processed = set()
    for fp in glob.glob(os.path.join(out_dir, "*.jsonl")):
        stem = os.path.splitext(os.path.basename(fp))[0]
        processed.add(stem)
    return processed


def process_single_pair(qa_pair, table_meta_infos, layered_tree, table_id, model_name, raw2subtab=None, invoke=False):
    cur_question = qa_pair["query"]

    # NEW table metadata format (labels only)
    # cur_table_meta = {
    #     "subtable_titles": table_meta_infos[table_id][0].get("subtable_titles", []),
    #     "column_headers": table_meta_infos[table_id][0].get("column_headers", []),
    #     "row_headers": table_meta_infos[table_id][0].get("row_headers", []),
    # }
    cur_table_meta = build_union_meta_for_raw(table_id, raw2subtab, table_meta_infos)

    start_time = time.time()

    # ---------- Stage 1: identify underspecified phrases + grounding ----------
    prompt1 = UncertaintyIdentifyAndGround_prompt.format(
        question=cur_question,
        table_metadata=json.dumps(cur_table_meta, ensure_ascii=False),
    )
    out1 = llm_generate(prompt1, model=model_name, json_format=True)

    input_tokens_1 = out1.get("input_tokens", 0)
    output_tokens_1 = out1.get("output_tokens", 0)

    try:
        res1 = extract_json_from_text(out1["text"])
    except Exception:
        res1 = {"underspecified_phrases": [], "selected_groundings": []}

    stage1_time = time.time() - start_time

    output = prune_phrase_groundings_with_layered_tree(layered_tree, res1['phrase_groundings'])

    underspecified_phrases, structural_elements = output[0], output[1]

    # If no underspecified phrases -> no rewrite
    if not underspecified_phrases:
        return {
            "ambiguity": False,
            "time_cost": stage1_time,
            "token_cost": [input_tokens_1, output_tokens_1],
            "underspecified_phrases": [],
            "selected_groundings": [],
            "rewritten_question": cur_question,
        }

    # ---------- Stage 2: rewrite question using pruned groundings ----------
    prompt2 = UncertaintyRewrite_prompt.format(
        question=cur_question,
        key_phrases=underspecified_phrases,
        structural_elements=structural_elements
    )
    out2 = llm_generate(prompt2, model=model_name, json_format=True)
    res2 = extract_json_from_text(out2["text"])

    input_tokens_2 = out2.get("input_tokens", 0)
    output_tokens_2 = out2.get("output_tokens", 0)

    rewritten = res2.get("rewritten_question", cur_question)

    total_time = time.time() - start_time
    total_in = input_tokens_1 + input_tokens_2
    total_out = output_tokens_1 + output_tokens_2

    return {
        "ambiguity": True,
        "time_cost": total_time,
        "token_cost": [total_in, total_out],
        # Stage-1 outputs
        "underspecified_phrases": underspecified_phrases,
        "selected_groundings": structural_elements,
        # Stage-2 outputs
        "rewritten_question": rewritten
    }


def run_multihiertt_benchmark(mode, model_name, cfg, args, aug_mode=False, overwrite_existing=True, qa_rids=[]):
    base_dir = cfg["base_folder"]
    out_folder = cfg["out_folder"]
    ds_cfg = cfg["datasets"][args.dataset]

    question_type = "single_tab"
    vlm_dir = os.path.join(out_folder, args.dataset, "our/table_processed/meta_infos")

    if not aug_mode:
        qa_path = os.path.join(base_dir, args.dataset, ds_cfg["qa_file"])
    else:
        qa_path = os.path.join(base_dir, args.dataset, ds_cfg["aug_qa_file"])

    out_dir = os.path.join(out_folder, args.dataset, ds_cfg["diamb_dir"], model_name)
    os.makedirs(out_dir, exist_ok=True)

    processed_ids = load_processed_ids(out_dir)

    print(f"🚀 [INFO] Starting Benchmark: {args.dataset}")

    # ---- load layered-tree label meta (keyed by SUBTABLE_ID) ----
    value_index_root = os.path.join(out_folder, args.dataset, "our/value_index")
    table_meta_infos, layered_tree = load_table_meta_from_layered_tree(value_index_root)
    print(f"[INFO] Loaded layered-tree meta for {len(table_meta_infos)} subtables")


    # ---- build RAW -> SUBTAB fallback mapping ----
    raw2subtab = {}

    multitab_path = os.path.join(out_folder, args.dataset, "our/table_processed/multitab_mapping.jsonl")
    if os.path.exists(multitab_path):
        with open(multitab_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_id = record.get("raw_table_id")
                subtab_ids = record.get("subtab_ids", []) or []
                # keep only those we actually have in table_meta_infos
                subtab_ids = [s for s in subtab_ids if s in table_meta_infos]
                if raw_id and subtab_ids:
                    raw2subtab[raw_id] = subtab_ids

    # also add a weak fallback from subtab_id naming: infer raw_id -> one subtab
    for sid in table_meta_infos.keys():
        if isinstance(sid, str) and sid.startswith("subtab_"):
            # infer raw from "subtab_{raw}_..."
            parts = sid.split("_", 2)
            if len(parts) >= 2:
                raw = parts[1]
                raw2subtab.setdefault(raw, [])
                if sid not in raw2subtab[raw]:
                    raw2subtab[raw].append(sid)

    def normalize_table_ids(tids):
        """
        Convert QA table_id(s) to subtab ids that exist in table_meta_infos.
        Strategy:
          - if tid is already a subtab id and exists -> keep
          - else if tid is a raw id and we have mapping -> replace with first mapped subtab
          - else drop
        Returns: list[subtab_id]
        """
        if isinstance(tids, str):
            tids = [tids]
        if not isinstance(tids, list):
            return []
        

        out = []
        for tid in tids:
            if tid in table_meta_infos:
                out.append(tid)
            elif tid in raw2subtab and raw2subtab[tid]:
                # pick first as default representative
                out.append(raw2subtab[tid][0])
            else:
                # unknown
                pass
        return out

    print(f"[INFO] Loading QA pairs from: {qa_path}")
    qa_pairs = read_jsonl(qa_path)

    if aug_mode:
        print(f"[INFO] Loaded {len(qa_pairs)} QA pairs")
        qa_pairs = [_ for _ in qa_pairs if _["llm"] == model_name]
        print(f"[Filtering] Resulting {len(qa_pairs)} QA pairs")

    print(f"[INFO] Loaded {len(qa_pairs)} QA pairs")

    valid_qa_pairs: List[Dict[str, Any]] = []
    print("[INFO] Filtering QA pairs...")

    dropped_no_meta = 0
    dropped_non_numeric = 0
    dropped_multi_table = 0

    for qa in tqdm(qa_pairs, total=len(qa_pairs)):
        tids_raw = qa.get("table_id")
        tids = normalize_table_ids(tids_raw)

        if not tids:
            dropped_no_meta += 1
            continue

        # update qa in-place so later code uses normalized subtab id
        qa["table_id"] = tids

        # numeric answer only
        try:
            float(qa["label"])
        except Exception:
            dropped_non_numeric += 1
            continue

        # single table only
        if question_type == "single_tab" and len(tids) > 1:
            dropped_multi_table += 1
            continue

        # VLM meta check is optional now; do NOT fail if missing
        # (and you currently don't even use single_subtable filter)
        try:
            _ = [json.load(open(os.path.join(vlm_dir, f"{tid}.json"), "r")) for tid in tids]
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[Filter Error] during VLM check: {e}")
            pass

        valid_qa_pairs.append(qa)

    print(f"[INFO] Dropped (no meta after normalize): {dropped_no_meta}")
    print(f"[INFO] Dropped (non-numeric label): {dropped_non_numeric}")
    print(f"[INFO] Dropped (multi-table for single_tab): {dropped_multi_table}")

    if not valid_qa_pairs:
        # provide actionable debugging info
        some_keys = list(table_meta_infos.keys())[:5]
        raise RuntimeError(
            "No valid QA pairs found after filtering.\n"
            f"- table_meta_infos size: {len(table_meta_infos)}\n"
            f"- example meta keys: {some_keys}\n"
            f"- dropped_no_meta={dropped_no_meta}, dropped_non_numeric={dropped_non_numeric}, dropped_multi_table={dropped_multi_table}\n"
            "Likely cause: QA table_id uses raw ids but value_index/layered_tree was generated only for subtables.\n"
        )

    print(f"✅ [INFO] Found {len(valid_qa_pairs)} valid QA pairs after filtering.")
    failed_cases = []

    for qa_pair in tqdm(valid_qa_pairs, total=len(valid_qa_pairs)):
        table_id = qa_pair["table_id"][0]  # now normalized to subtab id

        if args.dataset == "realhitbench_num":
            processed_record = process_single_pair(qa_pair, table_meta_infos, layered_tree, tids_raw, model_name, raw2subtab, invoke=True)
        else:
            processed_record = process_single_pair(qa_pair, table_meta_infos, layered_tree, tids_raw, model_name, raw2subtab)

        save_fpath = os.path.join(out_dir, f'{qa_pair["id"]}.jsonl')
        with open(save_fpath, "w", encoding="utf-8") as f:
            f.write(json.dumps(processed_record, ensure_ascii=False, indent=2) + "\n")

    
    print(failed_cases)


def parse_option():
    parser = argparse.ArgumentParser("command line arguments for generation.")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument(
        "--mode",
        type=str,
        default="raw"
    )
    parser.add_argument("--model", type=str, default="GPT-4o")
    parser.add_argument("--aug", action="store_true")
    parser.add_argument(
        "--qa_rids",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=[],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_option()

    model_name = MODEL_MAP[args.model]

    with open("../config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    if args.dataset == "multihiertt_num":
        args.qa_rids = [int(_) for _ in args.qa_rids]

    run_multihiertt_benchmark(
        args.mode,
        model_name,
        cfg,
        args,
        aug_mode=args.aug,
        overwrite_existing=True,
        qa_rids=args.qa_rids,
    )