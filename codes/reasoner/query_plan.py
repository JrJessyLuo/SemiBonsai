# -*- coding: utf-8 -*-

import os
import json
import time
import pickle
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy

import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

from utils.basic_utils import (
    analyze_llm_output,
    load_multitab_mapping_jsonl,
    extract_python_code,
    run_extracted_code,
    load_table_meta,
    read_jsonl,
    extract_json_from_text,
    parse_llm_dict,
    safe_json_loads,
    load_table_meta_from_layered_tree,
    embed_fn,   # optional
)

from utils.api_utils import llm_generate_setup as llm_generate
from utils.constants import MODEL_MAP

from operation import (
    action_generation_prompt,
    program_generation_prompt,
    schema_linking_prompt,
)

LLM_MODEL = "gpt-4.1"
TEMPERATURE = 0


# -----------------------------------------------------------------------------
# Value-index helpers (NEW)
# -----------------------------------------------------------------------------
def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _load_layered_tree(value_index_dir: str) -> Dict[str, Any]:
    p = os.path.join(value_index_dir, "layered_tree.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing layered_tree.json: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_subtables_from_tree(layered_tree: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Return list of {"subtable_id":..., "title":...}
    """
    out = []
    for n in layered_tree.get("nodes", []):
        if n.get("type") == "subtable":
            sid = str(n.get("subtable_id", "")).strip()
            title = str(n.get("title", "")).strip()
            if sid:
                out.append({"subtable_id": sid, "title": title})
    # stable sort by title then id
    out.sort(key=lambda x: (x.get("title", ""), x.get("subtable_id", "")))
    return out


def _normalize_text(s: str) -> List[str]:
    import re
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = [t for t in s.split() if t]
    return toks


def choose_best_subtable(
    query: str,
    subtables: List[Dict[str, str]],
    *,
    use_embed: bool = False,
    embed_fn_callable=None,
) -> Optional[Dict[str, str]]:
    """
    Pick best subtable by:
      - if use_embed and embed_fn_callable: cosine similarity between query and title
      - else token-overlap score between query tokens and title tokens
    """
    if not subtables:
        return None
    if len(subtables) == 1:
        return subtables[0]

    # ---- embedding route ----
    if use_embed and embed_fn_callable is not None:
        try:
            q_emb = embed_fn_callable([query])[0]
            titles = [st.get("title", "") or st.get("subtable_id", "") for st in subtables]
            t_embs = embed_fn_callable(titles)

            # cosine
            def cos(a, b):
                denom = (np.linalg.norm(a) * np.linalg.norm(b))
                if denom == 0:
                    return -1.0
                return float(np.dot(a, b) / denom)

            best_idx, best_score = 0, -1e9
            for i, emb in enumerate(t_embs):
                score = cos(q_emb, emb)
                if score > best_score:
                    best_idx, best_score = i, score
            return subtables[best_idx]
        except Exception:
            # fallback to token overlap
            pass

    # ---- token overlap fallback ----
    q_toks = set(_normalize_text(query))
    best = None
    best_score = -1
    for st in subtables:
        title = st.get("title", "") or st.get("subtable_id", "")
        t_toks = set(_normalize_text(title))
        if not t_toks:
            score = 0
        else:
            score = len(q_toks & t_toks)
        if score > best_score:
            best_score = score
            best = st
    return best


def build_df_for_subtable(
    value_index_dir: str,
    subtable_id: str,
    *,
    relevant_row_headers: Optional[List[str]] = None,
    relevant_column_headers: Optional[List[str]] = None,
    na_value: Any = "",
) -> pd.DataFrame:
    """
    Construct a dataframe for ONE subtable_id from raw-level:
      mapping_table.csv: col_path_id,row_path_id,value_id
      value_table.csv: value_id,value

    IMPORTANT:
    - We first filter mapping_table to only rows whose col_path_id and row_path_id belong to `subtable_id`.
    - If relevant_row_headers/column_headers are provided, we further filter by leaf labels using layered_tree.json.

    Output index/columns are "pretty labels" built from layered_tree:
      index:  "<title> | <row-path...>"
      col:    "<title> | <col-path...>"
    """
    layered_tree = _load_layered_tree(value_index_dir)
    nodes = layered_tree.get("nodes", [])
    # path_id -> (type, path(list), label, subtable_id, subtable_title)
    pid2info: Dict[str, Dict[str, Any]] = {}

    # build subtable node mapping
    stid2title: Dict[str, str] = {}
    for n in nodes:
        if n.get("type") == "subtable":
            sid = str(n.get("subtable_id", "")).strip()
            stid2title[sid] = str(n.get("title", "")).strip()

    for n in nodes:
        if n.get("type") in ("colhdr", "rowhdr") and n.get("is_leaf", False):
            pid = str(n.get("path_id", "")).strip()
            if not pid:
                continue
            path = n.get("path")
            if not isinstance(path, list) or not path:
                # fallback to single label
                lab = str(n.get("label", "")).strip()
                path = [lab] if lab else []
            # pid contains prefix: subtab_xxx::C::... or subtab_xxx::R::...
            # we can parse subtable_id from pid directly (safe)
            sid = pid.split("::", 1)[0]
            pid2info[pid] = {
                "type": n.get("type"),
                "path": [str(x).strip() for x in path if str(x).strip()],
                "label": str(n.get("label", "")).strip(),
                "subtable_id": sid,
                "subtable_title": stid2title.get(sid, ""),
            }

    mapping_path = os.path.join(value_index_dir, "mapping_table.csv")
    value_path = os.path.join(value_index_dir, "value_table.csv")
    mp = _safe_read_csv(mapping_path)
    vt = _safe_read_csv(value_path)

    # expected columns
    for col in ["col_path_id", "row_path_id", "value_id"]:
        if col not in mp.columns:
            raise ValueError(f"mapping_table.csv missing column {col}. columns={list(mp.columns)}")
    for col in ["value_id", "value"]:
        if col not in vt.columns:
            raise ValueError(f"value_table.csv missing column {col}. columns={list(vt.columns)}")

    mp["col_path_id"] = mp["col_path_id"].astype(str)
    mp["row_path_id"] = mp["row_path_id"].astype(str)

    # 1) restrict to this subtable_id
    # path_id prefix is subtable_id
    mp_sub = mp[
        mp["col_path_id"].str.startswith(subtable_id + "::")
        & mp["row_path_id"].str.startswith(subtable_id + "::")
    ].copy()

    # 2) optional further filtering by header LABELS (leaf labels)
    #    Here we accept "Muslim" or "Degree or Equivalent Estimate" etc.
    if relevant_row_headers:
        wanted = set([str(x).strip() for x in relevant_row_headers if str(x).strip()])
        if wanted:
            keep_row_pids = []
            for pid in mp_sub["row_path_id"].unique().tolist():
                info = pid2info.get(pid)
                if not info:
                    continue
                # match by leaf label OR full leaf path string
                leaf_label = (info.get("label") or "").strip()
                leaf_path_str = " ".join(info.get("path") or [])
                if leaf_label in wanted or leaf_path_str in wanted:
                    keep_row_pids.append(pid)
            if keep_row_pids:
                mp_sub = mp_sub[mp_sub["row_path_id"].isin(keep_row_pids)]

    if relevant_column_headers:
        wanted = set([str(x).strip() for x in relevant_column_headers if str(x).strip()])
        if wanted:
            keep_col_pids = []
            for pid in mp_sub["col_path_id"].unique().tolist():
                info = pid2info.get(pid)
                if not info:
                    continue
                leaf_label = (info.get("label") or "").strip()
                leaf_path_str = " ".join(info.get("path") or [])
                if leaf_label in wanted or leaf_path_str in wanted:
                    keep_col_pids.append(pid)
            if keep_col_pids:
                mp_sub = mp_sub[mp_sub["col_path_id"].isin(keep_col_pids)]

    # join values
    merged = mp_sub.merge(vt, on="value_id", how="left")
    merged["value"] = merged["value"].fillna(na_value).astype(str)

    # pivot
    pivot = merged.pivot_table(
        index="row_path_id",
        columns="col_path_id",
        values="value",
        aggfunc="first",
        dropna=False,
    )

    # pretty labels
    def pretty(pid: str) -> str:
        info = pid2info.get(pid)
        if not info:
            return pid
        st_title = info.get("subtable_title", "")
        chain = info.get("path") or []
        parts = []
        if st_title:
            parts.append(st_title)
        parts.extend(chain)
        return " | ".join([p for p in parts if p])

    pivot.index = [pretty(x) for x in pivot.index.tolist()]
    pivot.columns = [pretty(x) for x in pivot.columns.tolist()]
    pivot = pivot.fillna(na_value)
    return pivot


# -----------------------------------------------------------------------------
# Original pipeline code (mostly unchanged)
# -----------------------------------------------------------------------------
@dataclass
class TableCtx:
    question: str = ""
    table_meta: Dict[str, Any] = field(default_factory=lambda: {"column_headers": [], "row_headers": []})
    table_values: Optional[pd.DataFrame] = None

    subquestions: Dict[str, Any] = field(default_factory=dict)
    atomic_subquestions: Dict[str, Any] = field(default_factory=dict)
    atomic_subquestion_mapping: Dict[str, Any] = field(default_factory=dict)

    operation_history: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_history: List[Dict[str, Any]] = field(default_factory=list)
    last_operation: str = ""

    atomic_subquestion_programs: Dict[str, Any] = field(default_factory=dict)


def init_context(question: str, table_meta: Dict[str, Any], df: pd.DataFrame):
    ctx = TableCtx()
    ctx.question = question
    ctx.table_meta = table_meta
    ctx.table_values = df

    ctx.subquestions = {"raw": question}
    ctx.atomic_subquestions = {"raw": question}
    ctx.atomic_subquestion_mapping = {}

    ctx.last_operation = "init"
    ctx.operation_history = []
    ctx.reasoning_history = []
    return ctx


def update_context(ctx, action_output, function):
    source_qs = action_output["parameters"]["subq_id"]
    target_qs = action_output["parameters"]["subquestions"]

    ctx.subquestions.update(target_qs)

    if source_qs in ctx.atomic_subquestions:
        del ctx.atomic_subquestions[source_qs]
    ctx.atomic_subquestions.update(target_qs)

    ctx.last_operation = action_output["action"]
    ctx.operation_history.append(function)

    if ctx.last_operation == "infer_calculation_formula":
        formula = action_output["parameters"]["formula"]
        ctx.reasoning_history.append(f"To answer {source_qs}, you should utilize the formula: {formula}.")
    elif ctx.last_operation == "multihop_question_decomposition":
        order = action_output["parameters"]["order"]
        ctx.reasoning_history.append(
            f"To answer the complex question {source_qs}, you should answer simpler subquestions in this sequence: {order}."
        )
    return ctx


possible_next_operation_dict = {
    "init": ["infer_calculation_formula", "multihop_question_decomposition", "generate_execute_program"],
    "infer_calculation_formula": ["multihop_question_decomposition", "generate_execute_program"],
    "multihop_question_decomposition": ["infer_calculation_formula", "generate_execute_program"],
}


def question_decomposition(opeartion_set, q, meta, df, model_name, mode="raw", max_steps=6, skip=False):
    ctx = init_context(q, meta, df)
    total_input_tokens, total_output_tokens = 0, 0

    if skip:
        return ctx, total_input_tokens, total_output_tokens

    steps = 1
    while steps < max_steps:
        if mode in ["raw"]:
            prompt = action_generation_prompt.format(
                operator_set=opeartion_set[0],
                dataset_specific_example=opeartion_set[1],
                subquestions=json.dumps(ctx.atomic_subquestions, ensure_ascii=False),
                table_metadata=json.dumps(ctx.table_meta, ensure_ascii=False),
                operation_history=json.dumps(ctx.operation_history, ensure_ascii=False),
                possible_actions=possible_next_operation_dict[ctx.last_operation],
            )

        if "deepseek" in model_name.lower():
            all_output = llm_generate(prompt, model=model_name, json_format=True)
            try:
                output = extract_json_from_text(all_output["text"])
            except Exception:
                print("-------------------------")
                print(all_output["text"])
                return None, 0, 0
        else:
            for _ in range(3):
                all_output = llm_generate(prompt, model=model_name, json_format=True)
                try:
                    output = parse_llm_dict(all_output["text"])
                    break
                except Exception:
                    prompt = prompt + "\n\nReturn ONLY a complete JSON object (no markdown fences)."

        input_tokens, output_tokens = all_output["input_tokens"], all_output["output_tokens"]
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        if "function" not in output:
            break

        function = output["function"]
        if "generate_execute_program" in function:
            break

        try:
            action_output = analyze_llm_output(function)
            prev_ctx = deepcopy(ctx)
            steps += 1
            ctx = update_context(ctx, action_output, function)
        except Exception as e:
            print("---error ", e)
            ctx = prev_ctx

    return ctx, total_input_tokens, total_output_tokens


def schema_linking(ctx, model_name):
    prompt = schema_linking_prompt.format(
        subquestions=json.dumps(ctx.atomic_subquestions, ensure_ascii=False),
        table_metadata=json.dumps(ctx.table_meta, ensure_ascii=False),
    )

    max_retries = 3
    input_tokens, output_tokens = 0, 0
    output = ""

    for attempt in range(max_retries):
        try:
            print(f"[INFO] Attempt {attempt + 1}/{max_retries} for schema linking.")
            all_output = llm_generate(prompt, model=model_name, json_format=True)
            output = all_output["text"]
            input_tokens, output_tokens = all_output["input_tokens"], all_output["output_tokens"]

            subq_matched_schema = safe_json_loads(output)
            ctx.atomic_subquestion_mapping = subq_matched_schema
            break
        except Exception as e:
            print(output)
            print("--errr", e)

    return input_tokens, output_tokens


def program_compose(ctx, subq_relevant_data, model_name, numerical_reasoning_context=""):
    prompt = program_generation_prompt.format(
        table=ctx.table_values,
        atomic_subquestions=json.dumps(ctx.atomic_subquestions, ensure_ascii=False),
        atomic_subquestions_subdata=json.dumps(subq_relevant_data, ensure_ascii=False),
        subquestions=json.dumps(
            {k: v for k, v in ctx.subquestions.items() if k not in ctx.atomic_subquestions.keys()}, ensure_ascii=False
        ),
        reasoning_history=ctx.reasoning_history,
        numerical_reasoning_context=numerical_reasoning_context,
    )

    all_output = llm_generate(prompt, model=model_name)

    output = all_output["text"]
    input_tokens, output_tokens = all_output["input_tokens"], all_output["output_tokens"]

    generated_python_code = extract_python_code(output)
    final_ans = run_extracted_code(generated_python_code)
    return generated_python_code, final_ans, input_tokens, output_tokens


def make_json_safe(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.generic):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj.item()
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {make_json_safe(k): make_json_safe(v) for k, v in obj.items()}
    return str(obj)


def process_qa_pair(
    opeartion_set,
    qa_pair: Dict[str, Any],
    table_meta_infos: Dict[str, Any],
    value_index_root: str,   # NEW
    out_dir: str,
    diamb_dir: str,
    overwrite_existing: bool,
    question_type: str,
    mode: str,
    model_name: str,
    use_embed_for_subtab: bool = False,
):
    qa_id = f"{question_type}_{qa_pair['id']}"
    case_dir = os.path.join(out_dir, qa_id)
    os.makedirs(case_dir, exist_ok=True)

    result_path = os.path.join(case_dir, "result.json")
    action_path = os.path.join(case_dir, "query_plan.txt")

    if (not overwrite_existing) and os.path.isfile(result_path):
        try:
            old = json.load(open(result_path, "r", encoding="utf-8"))
            if isinstance(old, list) and len(old) > 0:
                print(f"[SKIP] {qa_id} already processed → {result_path}")
                return
        except Exception:
            pass

    # raw table id (value_index is stored at raw level)
    raw_table_id = qa_pair["table_id"][0] if type(qa_pair["table_id"])==list else qa_pair["table_id"]

    value_index_dir = os.path.join(value_index_root, raw_table_id)
    if not os.path.isdir(value_index_dir):
        print(f"[ERROR] value_index_dir not found: {value_index_dir}")
        return

    question = qa_pair["query"]

    # ---- pick best subtable from layered_tree.json ----
    try:
        layered_tree = _load_layered_tree(value_index_dir)
        subtables = _list_subtables_from_tree(layered_tree)
        best = choose_best_subtable(
            question,
            subtables,
            use_embed=use_embed_for_subtab,
            embed_fn_callable=embed_fn if use_embed_for_subtab else None,
        )
        if best is None:
            print(f"[ERROR] No subtable found in layered_tree for raw={raw_table_id}")
            return
        chosen_subtab_id = best["subtable_id"]
        chosen_subtab_title = best.get("title", "")
    except Exception as e:
        print(f"[ERROR] Failed to select subtable for {qa_id}: {e}")
        return

    print(table_meta_infos.keys())
    # ---- load meta for the chosen subtable id ----
    if chosen_subtab_id not in table_meta_infos:
        print(f"[ERROR] Missing meta for chosen subtable: {chosen_subtab_id}")
        return
    meta = table_meta_infos[chosen_subtab_id]
    meta_for_ctx = {
        "column_headers": meta[0].get("column_headers", []),
        "row_headers": meta[0].get("row_headers", []),
    }

    # ---- Load full df for that chosen subtable ----
    try:
        df = build_df_for_subtable(value_index_dir, chosen_subtab_id)

        print(f"[START] Processing {qa_id} (raw={raw_table_id}, chosen_subtab={chosen_subtab_id}, title={chosen_subtab_title})")
    except Exception as e:
        print(f"[ERROR] Failed to build df for {qa_id}: {e}")
        return

    # ---- NEW FORMAT ONLY: single JSON object with top-level rewritten_question ----
    disamb_path = os.path.join(diamb_dir, model_name, f"{qa_pair['id']}.jsonl")
    rewritten_q = ""
    if os.path.exists(disamb_path):
        try:
            with open(disamb_path, "r", encoding="utf-8") as f:
                disamb_candidates = json.load(f)
            if isinstance(disamb_candidates, dict):
                rewritten_q = str(disamb_candidates.get("rewritten_question", "")).strip()
        except Exception:
            rewritten_q = ""

    qa_pair_local = dict(qa_pair)
    if rewritten_q:
        qa_pair_local["query"] = rewritten_q

    # ---- Operation set ----
    current_operation_set = opeartion_set.copy()
    if len(opeartion_set[0].strip()) == 0:
        current_operation_set[0] = qa_pair_local.get("aggregation", "")

    # ---- Single run only ----
    ctx_dict, result_data = process_single_res(
        current_operation_set,
        qa_pair_local,
        qa_id,
        meta_for_ctx,
        df,
        value_index_dir,
        chosen_subtab_id,
        mode,
        model_name,
        max_retries=3,
    )

    with open(action_path, "w", encoding="utf-8") as f:
        json.dump([ctx_dict], f, indent=4, ensure_ascii=False)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump([result_data], f, indent=4, ensure_ascii=False)

    print(f"[DONE] Results saved to {result_path}")


def process_single_res(
    opeartion_set,
    qa_pair,
    qa_id,
    meta_for_ctx,
    df,
    value_index_dir: str,      # NEW
    chosen_subtab_id: str,     # NEW
    mode,
    model_name,
    max_retries=3,
):
    step1_in, step1_out = 0, 0
    question = qa_pair["query"]

    # ---- decomposition ----
    if mode == "remove_all":
        ctx, _, _ = question_decomposition(
            opeartion_set, question, meta_for_ctx, df, model_name, mode=mode, max_steps=6, skip=True
        )
        cost_time_decomp = 0
    else:
        ctx = None
        cost_time_decomp = 0
        for attempt in range(max_retries):
            try:
                print(f"[INFO] Attempt {attempt + 1}/{max_retries} for question decomposition.")
                start = time.time()

                for q_attempt in range(3):
                    ctx, cur_in, cur_out = question_decomposition(
                        opeartion_set, question, meta_for_ctx, df, model_name, mode=mode, max_steps=6
                    )
                    if ctx is not None:
                        break
                    cur_in, cur_out = 0, 0
                    start = time.time()
                    if q_attempt == 2:
                        return None, None

                step1_in += cur_in
                step1_out += cur_out
                cost_time_decomp = time.time() - start
                print(f"[INFO] Decomposition successful on attempt {attempt + 1}.")
                break
            except Exception:
                continue

    # ---- schema linking ----
    start_time_linking = time.time()
    step2_in, step2_out = 0, 0
   
    step2_in, step2_out = schema_linking(ctx, model_name)
    cost_time_linking = time.time() - start_time_linking
   

    ctx_dict = {
        "subquestions": ctx.subquestions,
        "atomic_subquestions": ctx.atomic_subquestions,
        "operation_history": ctx.operation_history,
        "reasoning_history": ctx.reasoning_history,
        "last_operation": ctx.last_operation,
        "schema_linking": ctx.atomic_subquestion_mapping,
    }

    # ---- build subdata for program generation (NOW via value_index + chosen subtable) ----

    atomic_subquestion_subdata = {}
    for subq, relevant_info in ctx.atomic_subquestion_mapping.items():
        if "relevant_row_headers" not in relevant_info or "relevant_column_headers" not in relevant_info:
            continue
        atomic_subquestion_subdata[subq] = build_df_for_subtable(
            value_index_dir,
            chosen_subtab_id,
            relevant_row_headers=relevant_info["relevant_row_headers"],
            relevant_column_headers=relevant_info["relevant_column_headers"],
        ).to_string(index=True)

    # ---- compose program ----
    start_time_compose = time.time()
    numerical_reasoning_context = qa_pair.get("aggregation", [])
    generated_python_code, final_ans, step3_in, step3_out = program_compose(
        ctx, atomic_subquestion_subdata, model_name, numerical_reasoning_context
    )
    cost_time_compose = time.time() - start_time_compose

    result_data = {
        "question_id": qa_id,
        "raw_table_id": qa_pair["table_id"][0] if type(qa_pair["table_id"])==list else qa_pair["table_id"],
        "chosen_subtable_id": chosen_subtab_id,
        "question": question,
        "label": qa_pair.get("label"),
        "final_answer": make_json_safe(final_ans),
        "generated_python_code": generated_python_code,
        "time_cost": {
            "decomposition": cost_time_decomp,
            "composition": cost_time_compose,
            "grounding": cost_time_linking,
            "total": cost_time_decomp + cost_time_compose + cost_time_linking,
        },
        "token_cost": {
            "decomposition": [step1_in, step1_out],
            "composition": [step3_in, step3_out],
            "grounding": [step2_in, step2_out],
            "total": [step1_in + step2_in + step3_in, step1_out + step2_out + step3_out],
        },
    }

    return ctx_dict, result_data


def run_multihiertt_benchmark(opeartion_set, mode, model_name, cfg, args, aug_mode=False, overwrite_existing=True):
    base_dir = cfg["base_folder"]
    out_folder = cfg["out_folder"]
    ds_cfg = cfg["datasets"][args.dataset]

    question_type = "single_tab"
    qa_path = os.path.join(base_dir, args.dataset, ds_cfg["qa_file"] if not aug_mode else ds_cfg["aug_qa_file"])
    diamb_dir = os.path.join(out_folder, args.dataset, ds_cfg["diamb_dir"])
    out_dir = os.path.join(out_folder, args.dataset, ds_cfg["result_dir"], f"runs_out_{mode}_{model_name}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"🚀 [INFO] Starting Benchmark: {args.dataset}")

    value_index_root = os.path.join(out_folder, args.dataset, "our/value_index")
    table_meta_infos,_ = load_table_meta_from_layered_tree(value_index_root)
    
    
    print(f"[INFO] Loaded meta for {len(table_meta_infos)} tables")

    print(f"[INFO] Loading QA pairs from: {qa_path}")
    qa_pairs = read_jsonl(qa_path)

    if aug_mode:
        qa_pairs = [x for x in qa_pairs if x.get("llm") == model_name]
        print(f"[Filtering] Resulting {len(qa_pairs)} QA pairs")

    # NOTE:
    # 这里不再 clean_qa_pairs 去改 table_id 为 subtab
    # 因为我们在 process_qa_pair 里会用 layered_tree.json 选择 best_subtable

    valid_qa_pairs: List[Dict[str, Any]] = []
    print("[INFO] Filtering QA pairs...")

    for qa in tqdm(qa_pairs, total=len(qa_pairs)):
        tids = qa.get("table_id")
        if isinstance(tids, str):
            tids = [tids]
        if not isinstance(tids, list) or not tids:
            continue

        raw_tid = tids[0]
        # 必须存在 value_index/raw_tid
        if not os.path.isdir(os.path.join(value_index_root, raw_tid)):
            continue

        # label must be numeric (your original constraint)
        try:
            float(qa.get("label"))
        except Exception:
            continue

        valid_qa_pairs.append(qa)

    if not valid_qa_pairs:
        raise RuntimeError("No valid QA pairs found after filtering. Adjust filters.")

    print(f"✅ [INFO] Found {len(valid_qa_pairs)} valid QA pairs after filtering.")

    failed_cases = []
    for qa_pair in tqdm(valid_qa_pairs, total=len(valid_qa_pairs)):
        # try:
        process_qa_pair(
            opeartion_set,
            qa_pair=qa_pair,
            table_meta_infos=table_meta_infos,
            value_index_root=value_index_root,
            out_dir=out_dir,
            diamb_dir=diamb_dir,
            overwrite_existing=overwrite_existing,
            question_type=question_type,
            mode=mode,
            model_name=model_name,
            use_embed_for_subtab=False,  # 你要 embedding 就改 True
        )
        # except Exception:
        #     failed_cases.append(qa_pair.get("id"))

    print("---", failed_cases)


def parse_option():
    parser = argparse.ArgumentParser("command line arguments for generation.")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument(
        "--mode",
        type=str,
        default="raw",
        choices=["raw"],
    )
    parser.add_argument("--model", type=str, default="GPT-4o")
    parser.add_argument("--aug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_option()
    model_name = MODEL_MAP[args.model]

    with open("../config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    if "hitab_num" in args.dataset:
        opeartion_set = [
            """
        - filter_tree, filter_level
        - sum, average, count, max, min
        - argmax, argmin, kth-argmax, kth-argmin, pair-argmax, pair-argmin
        - difference, proportion, difference_rate, opposite
        - greater_than, less_than, eq, not_eq
        """,
            """
        subquestions: {"raw": "What is the maximum sales in any region in 2020?"}
        table metadata: {"row_headers": ["Region A", "Region B", "Region C"], "column_headers": ["2020 Sales"]}
        action history: []
        Function:
        infer_calculation_formula(
            "raw",
            "max(#1, #2, #3)",
            {
            "#1": "Region A, 2020 Sales",
            "#2": "Region B, 2020 Sales",
            "#3": "Region C, 2020 Sales"
            }
        )
        Explanation: Use the max operator to select the largest value among the regions.
        """,
        ]
    elif "multihiertt_num" in args.dataset:
        opeartion_set = [
            """
        - Add (+), Subtract (−), Multiply (×), Divide (÷), Exp (^)
        """,
            """
        subquestions: {"raw": "What is the sum of revenue in 2018 and 2019?"}
        table metadata: {"column_headers": ["2017 Revenue", "2018 Revenue", "2019 Revenue"]}
        action history: []
        Function:
        infer_calculation_formula(
            "raw",
            "#1 + #2",
            {
            "#1": "2018 Revenue",
            "#2": "2019 Revenue"
            }
        )
        Explanation: The answer requires adding revenues from 2018 and 2019. Use + from the Operator Set.
        """,
        ]
    else:
        opeartion_set = [
            """
        Operator Set:
        - sum, count, sort, argmax, argmin, max, min
        - diff, average, divide, multiply, add, subtract
        - comparison (greater_than, less_than, equal)
        """,
            """
        subquestions: {"raw": "What is the difference in population between 2021 and 2019?"}
        table metadata: {"column_headers": ["2019 Population", "2021 Population"]}
        action history: []
        Function:
        infer_calculation_formula(
            "raw",
            "#2 - #1",
            {
            "#1": "2019 Population",
            "#2": "2021 Population"
            }
        )
        Explanation: Difference is calculated using the diff (-) operator from the Operator Set.
        """,
        ]

    run_multihiertt_benchmark(
        opeartion_set, args.mode, model_name, cfg, args, aug_mode=args.aug, overwrite_existing=True
    )