#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from utils.api_utils import llm_generate


# ============================================================
# 0) Prompt (LLM fallback)
# ============================================================
evaluation_prompt_num = """
You are a strict numeric comparator. Output ONLY "T" or "F".

Input:
A: {a}
B: {b}

Procedure:
1) Extract the first numeric value from A and B. If either is missing, output F.
   - Ignore surrounding text, commas, currency symbols, and percent signs.

2) Unit handling (optional):
   - If both contain a recognizable unit, convert to the same base unit BEFORE comparing.
   - If units are incompatible (different physical dimension), output F.
   - If one or both have no unit, compare as plain numbers.

   Supported conversions (examples, not exhaustive):
   - percent: X% = X * 0.01
   - thousand/million/billion/trillion: *1e3 / *1e6 / *1e9 / *1e12
   - 千/万/亿: *1e3 / *1e4 / *1e8
   - length: mm/cm/m/km -> *1e-3 / *1e-2 / *1 / *1e3
   - mass: mg/g/kg -> *1e-6 / *1e-3 / *1
   - time: ms/s/min/h -> *1e-3 / *1 / *60 / *3600

3) Let nA, nB be the normalized numbers.
   Output T if:
     |nA - nB| <= max(1e-6, 0.02 * max(|nA|, |nB|, 1.0))

4) If not equal by (3), allow common scale-factor mismatches:
   If nB/nA (or nA/nB) is within 2% of any factor in:
     {0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e6, 1e8, 1e9, 1e12}
   then output T.

Otherwise output F.
""".strip()


# ============================================================
# 1) Rule-based fair numeric compare (from your snippet, bug-fixed)
# ============================================================

def normalize_num_str(s: Any) -> Optional[float]:
    if s is None:
        return None

    s = str(s).strip()
    if not s:
        return None

    is_percent = "%" in s
    s = s.replace("%", "")

    # Keep only digits, separators, sign, exponent
    s = re.sub(r"[^0-9.,eE+\-]", "", s)
    if not s:
        return None

    # Decide decimal vs thousand separators
    if "," in s and "." in s:
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        if last_comma > last_dot:
            # comma = decimal, dots = thousands
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            # dot = decimal, commas = thousands
            s = s.replace(",", "")
    elif "," in s:
        # only comma -> treat as decimal
        s = s.replace(",", ".")

    try:
        val = float(Decimal(s))
    except (InvalidOperation, ValueError):
        return None

    if is_percent:
        val /= 100.0
    return val


def nearly_equal(
    a: float,
    b: float,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-8,
    allow_opposite_sign: bool = True,
) -> bool:
    # Standard near-equality
    diff = abs(a - b)
    scale = max(abs(a), abs(b), 1.0)
    if diff <= max(abs_tol, rel_tol * scale):
        return True

    # Optional: magnitude-equality even if sign differs
    if allow_opposite_sign:
        mag_diff = abs(abs(a) - abs(b))
        mag_scale = max(abs(a), abs(b), 1.0)
        if mag_diff <= max(abs_tol, rel_tol * mag_scale):
            return True

    return False


def _digits_signature(s: Any) -> str:
    ds = re.sub(r"\D", "", str(s))
    ds = ds.lstrip("0") or "0"
    return ds


def fair_compare_num_str(
    sa: Any,
    sb: Any,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-8,
    max_exp: int = 3,
) -> Tuple[bool, Optional[float], Optional[float], Optional[int]]:
    """
    Fair comparison between two numeric strings/values, allowing:
      - tolerance-based equality
      - percent/fraction (x100)
      - 10^k scaling for |k| <= max_exp, only if digit-signature matches

    Returns: (is_equal, va, vb, used_exp)
      used_exp:
        0 for direct
        +/-2 for percent-style (x100)
        k for 10^k scaling
        None for not equal
    """
    va = normalize_num_str(sa)
    vb = normalize_num_str(sb)
    if va is None or vb is None:
        return False, va, vb, None

    # 1) direct
    if nearly_equal(va, vb, rel_tol=rel_tol, abs_tol=abs_tol):
        return True, va, vb, 0

    # 2) percent/fraction (x100), no digit-signature constraint
    if nearly_equal(va * 100.0, vb, rel_tol=rel_tol, abs_tol=abs_tol):
        return True, va, vb, +2
    if nearly_equal(vb * 100.0, va, rel_tol=rel_tol, abs_tol=abs_tol):
        return True, va, vb, -2

    # 3) general 10^k scaling but only if digit patterns match
    sig_a = _digits_signature(sa)
    sig_b = _digits_signature(sb)
    if sig_a != sig_b:
        return False, va, vb, None

    for k in range(-max_exp, max_exp + 1):
        if k == 0:
            continue
        factor = 10.0 ** k
        if nearly_equal(va * factor, vb, rel_tol=rel_tol, abs_tol=abs_tol):
            return True, va, vb, k
        if nearly_equal(vb * factor, va, rel_tol=rel_tol, abs_tol=abs_tol):
            return True, va, vb, -k

    return False, va, vb, None


# ============================================================
# 2) LLM fallback comparator (your required logic)
# ============================================================

LLMGenerateFn = Callable[..., Any]  # expected signature: llm_generate(prompt, model=..., **kwargs)

def _normalize_llm_result(x: Any) -> str:
    """
    Your llm_generate sometimes returns dict with 'text'.
    Normalize to "T"/"F".
    """
    if isinstance(x, dict) and "text" in x:
        x = x["text"]
    s = str(x).strip().upper()
    return "T" if s.startswith("T") else "F"


def is_equal_num(
    a: Any,
    b: Any,
    *,
    llm_generate: Optional[LLMGenerateFn],
    llm_model: str = "gpt-5.1",
) -> str:
    """
    Two-stage:
      1) fair_compare_num_str
      2) LLM fallback if not matched
    Returns: "T" or "F"
    """
    is_correct, _, _, _ = fair_compare_num_str(a, b)
    if is_correct:
        return "T"

    if llm_generate is None:
        # If user didn't provide LLM, we cannot fallback
        return "F"

    prompt = evaluation_prompt_num.format(a=str(a), b=str(b))
    out = llm_generate(prompt, model=llm_model)
    return _normalize_llm_result(out)


# ============================================================
# 3) Public API: evaluate one pair
# ============================================================

@dataclass
class EvalResult:
    is_correct: bool
    method: str                 # "rule" or "llm"
    pred_val: Optional[float]
    gt_val: Optional[float]
    used_exp: Optional[int]
    llm_judge: Optional[str]    # "T"/"F"/None


def evaluate_pair(
    prediction: Any,
    ground_truth: Any,
    *,
    llm_generate: Optional[LLMGenerateFn] = None,
    llm_model: str = "gpt-5.1",
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-8,
    max_exp: int = 3,
    use_llm_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate one (prediction, ground_truth) pair.
    Returns a JSON-serializable dict with:
      is_correct, method, pred_val, gt_val, used_exp, llm_judge
    """
    is_correct, va, vb, used_exp = fair_compare_num_str(
        prediction, ground_truth, rel_tol=rel_tol, abs_tol=abs_tol, max_exp=max_exp
    )
    if is_correct:
        r = EvalResult(
            is_correct=True, method="rule",
            pred_val=va, gt_val=vb, used_exp=used_exp,
            llm_judge=None
        )
        return r.__dict__

    if not use_llm_fallback:
        r = EvalResult(
            is_correct=False, method="rule",
            pred_val=va, gt_val=vb, used_exp=used_exp,
            llm_judge=None
        )
        return r.__dict__

    judge = is_equal_num(
        prediction, ground_truth, llm_generate=llm_generate, llm_model=llm_model
    )
    r = EvalResult(
        is_correct=(judge == "T"),
        method="llm",
        pred_val=va,
        gt_val=vb,
        used_exp=used_exp,
        llm_judge=judge
    )
    return r.__dict__


# ============================================================
# 4) Optional: batch evaluate JSONL (resume-safe)
# ============================================================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_is_correct=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_ids(output_jsonl: str, id_field: str) -> set:
    done = set()
    if not os.path.exists(output_jsonl):
        return done
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if id_field in obj:
                    done.add(str(obj[id_field]))
            except Exception:
                continue
    return done



# ============================================================
# 5) CLI
# ============================================================
def main():
    # a, b = '0.16', '1,6'
    # a, b = '14.0', '14'
    # a, b = 0.167001, 16.7
    a, b = 6.0, 0.0
    res = evaluate_pair(a, b)
    print(res)

if __name__ == "__main__":
    main()