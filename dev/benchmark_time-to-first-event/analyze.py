"""Analyze results/runs.jsonl: latency stats per (provider, model, effort) group."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_runs(runs_path: Path):
    runs = []
    for line in runs_path.read_text().splitlines():
        line = line.strip()
        if line:
            runs.append(json.loads(line))
    return runs


def pctile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    frac = k - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def fmt(v):
    if v is None:
        return "  —  "
    return f"{v:5.2f}"


def group_stats(runs: list[dict]) -> list[dict]:
    """Group by (provider, model, effort)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["provider"], r["model"], r["effort"])].append(r)

    out = []
    for (provider, model, effort), rs_all in groups.items():
        # Exclude error trials from latency stats (they distort the distribution).
        rs = [r for r in rs_all if not r.get("error")]
        first_disp = [r["first_displayable_s"] for r in rs if r.get("first_displayable_s") is not None]
        total = [r["total_s"] for r in rs if r.get("total_s") is not None]
        reason_complete = [(r["t_reasoning_complete"] - r["t_start"])
                           for r in rs if r.get("t_reasoning_complete") is not None]
        first_text = [(r["t_first_text_delta"] - r["t_start"])
                      for r in rs if r.get("t_first_text_delta") is not None]
        errors = sum(1 for r in rs_all if r.get("error"))
        reasoned = sum(1 for r in rs if r.get("t_reasoning_complete") is not None)
        rsn_tok = [r["reasoning_tokens"] for r in rs if r.get("reasoning_tokens")]
        out_tok = [r["output_tokens"] for r in rs if r.get("output_tokens")]

        out.append({
            "provider": provider,
            "model": model,
            "effort": effort,
            "n": len(rs_all),
            "n_ok": len(rs),
            "errors": errors,
            "reasoned": reasoned,
            "first_disp_median": statistics.median(first_disp) if first_disp else None,
            "first_disp_min": min(first_disp) if first_disp else None,
            "first_disp_max": max(first_disp) if first_disp else None,
            "reason_complete_median": statistics.median(reason_complete) if reason_complete else None,
            "first_text_median": statistics.median(first_text) if first_text else None,
            "total_median": statistics.median(total) if total else None,
            "rsn_tok_median": statistics.median(rsn_tok) if rsn_tok else None,
            "out_tok_median": statistics.median(out_tok) if out_tok else None,
        })
    return out


def print_table(stats: list[dict]):
    ordered = sorted(stats, key=lambda s: (s["provider"], s["model"], s["effort"]))
    effort_order = {"off": 0, "none": 0, "low": 1, "medium": 2, "high": 3}
    ordered.sort(key=lambda s: (s["provider"], s["model"], effort_order.get(s["effort"], 99)))

    MODEL_RENAME = {"gemini-3-flash-preview": "3-flash-preview"}
    hdr = (f"{'provider':<9} {'model':<20} {'effort':<16} {'n':>2} "
           f"{'err':>3} {'thk':>3} "
           f"{'first_disp(med)':>15} {'[min':>6} {'max]':>6} "
           f"{'rsn_done(med)':>13} {'1st_txt(med)':>12} "
           f"{'total(med)':>10} {'rsn_tok':>7} {'out_tok':>7}")
    print(hdr)
    print("-" * len(hdr))

    prev_model = None
    for s in ordered:
        key = (s["provider"], s["model"])
        if prev_model is not None and prev_model != key:
            print()
        prev_model = key
        model_display = MODEL_RENAME.get(s["model"], s["model"])
        print(
            f"{s['provider']:<9} {model_display:<20} {s['effort']:<16} {s['n']:>2} "
            f"{s['errors']:>3} {s['reasoned']:>3} "
            f"{fmt(s['first_disp_median']):>15} "
            f"{fmt(s['first_disp_min']):>6} {fmt(s['first_disp_max']):>6} "
            f"{fmt(s['reason_complete_median']):>13} "
            f"{fmt(s['first_text_median']):>12} "
            f"{fmt(s['total_median']):>10} "
            f"{str(s['rsn_tok_median'] or '—'):>7} "
            f"{str(s['out_tok_median'] or '—'):>7}"
        )


def print_errors(runs):
    errs = [r for r in runs if r.get("error")]
    if not errs:
        return
    print("\n=== Errors ===")
    for r in errs:
        print(f"{r['provider']} {r['model']} effort={r['effort']} trial={r['trial_idx']}: {r['error'][:300]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=Path(__file__).parent,
                    help="Directory containing results/runs.jsonl.")
    args = ap.parse_args()

    runs_path = Path(args.workdir).resolve() / "results" / "runs.jsonl"
    runs = load_runs(runs_path)
    if not runs:
        print("No runs yet.")
        return
    print(f"Total trials: {len(runs)}\n")
    print("Legend: err=errors, thk=#trials that produced a thinking trace, all times in seconds.")
    print("first_disp = min(reasoning-complete, first-text-delta) from request start.\n")
    stats = group_stats(runs)
    print_table(stats)
    print_errors(runs)


if __name__ == "__main__":
    main()
