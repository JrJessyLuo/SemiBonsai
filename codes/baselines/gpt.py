import os
import json
from typing import Dict, Any, List, Optional, Tuple
from utils.api_utils import *
from process_file_baseline import read_jsonl
import time
import argparse
import tqdm
from tqdm import tqdm

# ---------- helpers ----------
def _resolve_assets_for_qapair(qa: Dict[str, Any], id_map: Dict[str, Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (image_path, json_path) for this QA pair using id_map.

    id_map structure we created earlier:
      id_map[orig_id] = {
        "new_table_id": str|None,
        "image": "/path/to/image.png",
        "json": "/path/to/schema.json"
      }

    The QA pairs may have:
      - table_id updated to new id (e.g., "10{orig}")
      - orig_table_id with the original id (we added that earlier)
    """
    # Prefer resolving via original table id (if present)
    if "orig_table_id" in qa and qa["orig_table_id"] in id_map:
        entry = id_map[qa["orig_table_id"]]
        return entry.get("image"), entry.get("json")

    # Otherwise, try direct lookup using current table_id as original key
    tid = str(qa.get("table_id"))
    if tid in id_map:
        entry = id_map[tid]
        return entry.get("image"), entry.get("json")

    # Finally, try reverse-lookup by matching new_table_id
    for orig_id, entry in id_map.items():
        if entry.get("new_table_id") == tid:
            return entry.get("image"), entry.get("json")

    return None, None


def build_vlm_prompt_for_question(question: str) -> str:
    """
    Keep the prompt simple and force a numeric answer if possible.
    """
    return (
        "You will be given a single TABLE IMAGE.\n"
        "Answer the user question using ONLY the table content.\n"
        "If the answer is numeric, return ONLY the number (no units, no extra text).\n"
        "If it is a percentage in the table, return the decimal form if shown as decimal; "
        "otherwise return exactly as it appears.\n\n"
        f"Question: {question}"
    )


def build_llm_prompt_for_json(question: str, table_json: Any) -> str:
    """
    Provide the table JSON and ask for a concise numeric answer.
    """
    table_text = json.dumps(table_json, ensure_ascii=False)
    return (
        "You are given a table serialized as JSON. Use ONLY this JSON to answer the question.\n"
        "If the answer is numeric, return ONLY the number (no units, no extra text).\n"
        "If no exact match is possible, infer via simple arithmetic (+,-,*,/).\n\n"
        f"Question: {question}\n\n"
        f"Table JSON:\n{table_text}"
    )

def build_llm_prompt_for_html(question: str, table_text: Any) -> str:
    """
    Provide the table JSON and ask for a concise numeric answer.
    """
    return (
        "You are given an HTML table and a question.\n"
        "Return only the final numerical answer (usually a single number), "
        "Do NOT explain, do NOT show any reasoning steps, and do NOT add any extra text.\n"
        "Do NOT include units, symbols, labels, or words—only the raw number.\n\n"
        f"HTML Table:\n{table_text}\n\n"
         f"Question: {question}\n\n"
        f"Answer:"
    )


def answer_with_vlm(question: str, image_path: str) -> Optional[str]:
    if not image_path or not os.path.exists(image_path):
        return None
    prompt = build_vlm_prompt_for_question(question)
    try:
        ans = vlm_generate(prompt=prompt, image=image_path)
        # Trim fences/spaces just in case
        return (ans or "").strip().strip("`")
    except Exception as e:
        print(f"[WARN] VLM failed for image {image_path}: {e}")
        return None


def answer_with_llm(question: str, json_path: str) -> Optional[str]:
    if not json_path or not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            table_json = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read JSON file {json_path}: {e}")
        return None

    prompt = build_llm_prompt_for_json(question, table_json)
    try:
        ans = llm_generate(prompt)
        return (ans or "").strip().strip("`")
        
       
        
    except Exception as e:
        print(f"[WARN] LLM failed for json {json_path}: {e}")
        return None

#         llm_generate_setup(
#     prompt: str,
#     model: str,
#     key=LLM_API_KEY,
#     url=LLM_API_URL,
#     max_tokens: int = 8192,
#     temperature: float = 0.3,
#     max_retries: int = 2,
# )

def answer_with_llm_html(question, html_table, args):
    start_time = time.time()
    prompt = build_llm_prompt_for_html(question, html_table)
    print('--', prompt)
    # print(args.model)
    try:
        # output_ans = llm_generate_setup(prompt, args.model)
        output_ans = llm_generate_setup(
                prompt,
                args.model,
                json_format=False
            )
        # print('--', output_ans)
        time_cost = time.time() - start_time

        ans, input_toks, output_toks = output_ans['text'], output_ans['input_tokens'], output_ans['output_tokens']

        return (ans or "").strip().strip("`"), [input_toks, output_toks], time_cost
    except Exception as e:
        print(f"[WARN] LLM failed for json {json_path}: {e}")
        return "-1", [0, 0], 0

def run_answer_pipeline(
    updated_qa_pairs: List[Dict[str, Any]],
    id_map: Dict[str, Dict[str, Any]],
    out_jsonl_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    For each QA pair, resolve image/json from id_map, ask:
      - VLM with image
      - LLM with json
    and collect answers.

    Output records format:
      {
        "id": <qa id>,
        "table_id": <current table_id>,
        "orig_table_id": <orig> (if present),
        "query": <question>,
        "vlm_answer": <string or None>,
        "llm_answer": <string or None>,
        "assets": {"image": "...", "json": "..."}
      }
    """
    results = []
    for qa in updated_qa_pairs:
        q = qa.get("question") or qa.get("query")  # support either key
        if not q:
            continue

        image_path, json_path = _resolve_assets_for_qapair(qa, id_map)
        vlm_ans = answer_with_vlm(q, image_path) if image_path else None
        llm_ans = answer_with_llm(q, json_path) if json_path else None

        rec = {
            "id": qa.get("id"),
            "table_id": qa.get("table_id"),
            "orig_table_id": qa.get("orig_table_id"),
            "query": q,
            "label": qa.get("label"),
            "target_present": qa.get("target_present"),
            "target_virtual": qa.get("target_virtual"),
            "multitab": qa.get("multitab"),
            "vlm_answer": vlm_ans,
            "llm_answer": llm_ans,
            "assets": {"image": image_path, "json": json_path}
        }
        results.append(rec)

    # Optionally save as JSONL
    if out_jsonl_path:
        os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)
        with open(out_jsonl_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results



# # -------------- example usage --------------
def parse_option():
    parser = argparse.ArgumentParser("command line arguments for generation.")
    
    parser.add_argument('--dataset_name', type=str, help='dataset name')
    parser.add_argument('--model', type=str, default='openai/gpt-5.5', help='model name')
    # API keys / base URLs should be configured via environment variables or utils.api_utils,
    # not hard-coded in this script.
    parser.add_argument('--format', type=str, default='html', choices=['csv', 'html', 'latex', 'markdown'], help='output format')
    opt = parser.parse_args()

    return opt

import re

def extract_last_number(text: str) -> str:
    # Matches integers or floats (with optional minus sign)
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else ""


if __name__ == "__main__":
    opt = parse_option()

    # ----- paths -----
    # set the path
    dataset_dir  = os.environ.get("DATASET_DIR", "/path/to/ST-Raptor-new_update/datasets")
    dataset_name = getattr(opt, "dataset_name", "hitabnum")
    output_root = os.environ.get("OUTPUT_DIR", "/path/to/ST-Raptor-new_update/result")
    output_dir   = os.path.join(output_root, dataset_name, opt.model)
    os.makedirs(output_dir, exist_ok=True)
    print(opt.model)

    table_dir = os.path.join(dataset_dir, dataset_name, "data/html")
    qa_fpath  = os.path.join(dataset_dir, dataset_name, "data/single_tab_qa.jsonl")
    out_fpath = os.path.join(output_dir, "predictions.jsonl")
    processed_keys = set()


    # ----- load questions -----
    with open(qa_fpath, "r", encoding="utf-8") as fh:
        qa_items = [json.loads(line) for line in fh if line.strip()]


    outputs = []

    for item in tqdm(qa_items, total=len(qa_items)):
        # IDs & text
        table_id = (item.get("table_id") or [None])[0] if isinstance(item.get("table_id"), list) else item.get("table_id")
        question = item.get("query") or item.get("question")
        label    = item.get("label")
        question_id = item.get("question_id") or item.get("qa_id") or item.get("id")


        if not (table_id and question_id):
            print(f"[WARN] skip malformed item: {item}")
            continue

        question_key = str(question_id)
        if question_key in processed_keys:continue

        # pick table file (json preferred)
        json_path = os.path.join(table_dir, f"{table_id}.json")
        html_path = os.path.join(table_dir, f"{table_id}.html")
        table_fpath = json_path if os.path.exists(json_path) else (html_path if os.path.exists(html_path) else None)
        if not table_fpath:
            print(f"[WARN] table file missing for {table_id}")
            continue

        with open(table_fpath, "r", encoding="utf-8") as f:
            table_content = f.read()

        final_answer, token_cost, time_cost  = answer_with_llm_html(question, table_content, opt)

        rec = {
            "question_id":  question_id,
            "table_id":     table_id,
            "question":     question,
            "label":        label,
            "final_answer": final_answer,
            "time_cost":    {"total":time_cost},   # {"total": seconds, ...zeros}
            "token_cost":  {"total":token_cost},  # {"total":[in_tokens,out_tokens], ...zeros}
        }
        outputs.append(rec)
      