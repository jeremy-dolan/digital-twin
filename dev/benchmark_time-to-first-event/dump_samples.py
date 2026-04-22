"""Dump every trial's output to results/outputs/<model>/<effort>/trial_<n>.md."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def _opt(v):
    return f"{v:.2f}s" if v is not None else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=Path(__file__).parent,
                    help="Directory containing results/runs.jsonl; outputs/ is created under results/.")
    args = ap.parse_args()

    work_dir = Path(args.workdir).resolve()
    runs_path = work_dir / "results" / "runs.jsonl"
    out_dir = work_dir / "results" / "outputs"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    runs = [json.loads(l) for l in runs_path.read_text().splitlines() if l.strip()]

    # Group trials by (model, effort); assign sequential indices by t_start.
    by_me = defaultdict(list)
    for r in runs:
        by_me[(r["model"], r["effort"])].append(r)

    written = 0
    for (model, effort), trials in by_me.items():
        trials.sort(key=lambda r: r["t_start"])
        dest = out_dir / model / effort
        dest.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(trials):
            ts = r.get("t_start")
            rc = r.get("t_reasoning_complete")
            ft = r.get("t_first_text_delta")
            lines = [
                f"# {model} / effort={effort} / trial {i}",
                "",
                f"- provider: {r['provider']}",
                f"- original trial_idx: {r['trial_idx']}",
                f"- t_start (perf_counter): {ts:.2f}" if ts else "- t_start: —",
                f"- first_displayable: {_opt(r.get('first_displayable_s'))}",
                f"- reasoning_complete: {_opt((rc - ts) if rc and ts else None)}",
                f"- first_text_delta: {_opt((ft - ts) if ft and ts else None)}",
                f"- total: {_opt(r.get('total_s'))}",
                f"- input_tokens: {r.get('input_tokens')}",
                f"- output_tokens: {r.get('output_tokens')}",
                f"- reasoning_tokens: {r.get('reasoning_tokens')}",
            ]
            if r.get("error"):
                lines += ["", f"**ERROR:** {r['error']}"]

            if r.get("reasoning_text"):
                lines += ["", "## Reasoning trace", "", "```", r["reasoning_text"].rstrip(), "```"]

            if r.get("response_text"):
                lines += ["", "## Response", "", r["response_text"].rstrip()]

            (dest / f"trial_{i}.md").write_text("\n".join(lines) + "\n")
            written += 1

    print(f"Wrote {written} trial files under {out_dir}")
    for model_dir in sorted(out_dir.iterdir()):
        for effort_dir in sorted(model_dir.iterdir()):
            count = len(list(effort_dir.glob("trial_*.md")))
            print(f"  {model_dir.name}/{effort_dir.name}/  ({count} trials)")


if __name__ == "__main__":
    main()
