"""Latency benchmark: OpenAI Responses + Anthropic Messages + Google GenAI, streaming.

Measure time-to-first-user-displayable-event for a given prompt.
A displayable event is either:
  A) a reasoning trace completed (for pass-through to a summarizer), or
  B) the first output text delta (direct stream to the user).

Appends per-trial JSON record to results/runs.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import anthropic
import openai
from google import genai
from google.genai import types as genai_types

MAX_TOKENS = 16000


def load_env():
    """Manual .env loader (no extra dep)."""
    env_path = Path(__file__).parent / ".env"
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), val)


def load_prompt(prompt_path: Path) -> list[dict[str, str]]:
    msgs = []
    for line in prompt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        msgs.append(ast.literal_eval(line))
    return msgs


def openai_input(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
    # Responses API accepts developer/user roles directly.
    return [{"role": m["role"], "content": m["content"]} for m in msgs]


def anthropic_system_and_messages(msgs: list[dict[str, str]]):
    devs = [m["content"] for m in msgs if m["role"] == "developer"]
    users = [m for m in msgs if m["role"] == "user"]
    # Join developer messages as system blocks
    system = [{"type": "text", "text": d} for d in devs]
    messages = [{"role": "user", "content": users[0]["content"]}]
    return system, messages


def gemini_system_and_contents(msgs: list[dict[str, str]]):
    devs = [m["content"] for m in msgs if m["role"] == "developer"]
    users = [m for m in msgs if m["role"] == "user"]
    system_instruction = "\n\n".join(devs) if devs else None
    # For single-turn, contents can just be the user string.
    contents = users[0]["content"]
    return system_instruction, contents


@dataclass
class Trial:
    provider: str
    model: str
    effort: str
    trial_idx: int
    t_start: float = 0.0
    t_first_event: float | None = None
    t_first_reasoning_delta: float | None = None
    t_reasoning_complete: float | None = None
    t_first_text_delta: float | None = None
    t_complete: float | None = None
    reasoning_text: str = ""
    response_text: str = ""
    reasoning_summary_text_count: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    error: str | None = None
    # Summary
    first_displayable_s: float | None = None
    total_s: float | None = None

    def finalize(self):
        candidates = [t for t in (self.t_reasoning_complete, self.t_first_text_delta) if t is not None]
        if candidates:
            self.first_displayable_s = min(candidates) - self.t_start
        if self.t_complete is not None:
            self.total_s = self.t_complete - self.t_start


def run_openai_trial(client: openai.OpenAI, model: str, effort: str, trial_idx: int,
                     messages: list[dict], summary: str | None = None) -> Trial:
    # Compose an effort label that includes the summary variant so the
    # analysis groups them separately.
    label = effort if not summary else f"{effort}+{summary}"
    t = Trial(provider="openai", model=model, effort=label, trial_idx=trial_idx)

    kwargs = dict(
        model=model,
        input=openai_input(messages),
        stream=True,
        max_output_tokens=MAX_TOKENS,
    )
    reasoning: dict[str, Any] = {"effort": effort}
    if summary:
        reasoning["summary"] = summary
    kwargs["reasoning"] = reasoning

    t.t_start = time.perf_counter()
    saw_summary_text_done = False
    try:
        stream = client.responses.create(**kwargs)
        for event in stream:
            now = time.perf_counter()
            if t.t_first_event is None:
                t.t_first_event = now

            etype = getattr(event, "type", "")

            if etype == "response.reasoning_summary_text.delta":
                if t.t_first_reasoning_delta is None:
                    t.t_first_reasoning_delta = now
                delta = getattr(event, "delta", "")
                if delta:
                    t.reasoning_text += delta

            elif etype == "response.reasoning_summary_text.done":
                t.reasoning_summary_text_count += 1
                saw_summary_text_done = True

            elif etype == "response.output_item.done":
                item = getattr(event, "item", None)
                if item is not None and getattr(item, "type", "") == "reasoning":
                    if t.t_reasoning_complete is None:
                        t.t_reasoning_complete = now
                    # When reasoning.summary is set to concise/detailed, the full
                    # summary text only appears on output_item.done (not via deltas).
                    summary_parts = getattr(item, "summary", None) or []
                    if summary_parts and not t.reasoning_text:
                        chunks = []
                        for p in summary_parts:
                            ptext = getattr(p, "text", None)
                            if ptext:
                                chunks.append(ptext)
                        if chunks:
                            t.reasoning_text = "\n\n".join(chunks)
                    # Fallback: if no response.reasoning_summary_text.done
                    # events fired, count summary parts across every reasoning
                    # item (summed over all output_item.done events).
                    if not saw_summary_text_done:
                        t.reasoning_summary_text_count += len(summary_parts)

            elif etype == "response.output_text.delta":
                if t.t_first_text_delta is None:
                    t.t_first_text_delta = now
                delta = getattr(event, "delta", "")
                if delta:
                    t.response_text += delta

            elif etype == "response.completed":
                t.t_complete = now
                response = getattr(event, "response", None)
                usage = getattr(response, "usage", None) if response else None
                if usage is not None:
                    t.input_tokens = getattr(usage, "input_tokens", None)
                    t.output_tokens = getattr(usage, "output_tokens", None)
                    details = getattr(usage, "output_tokens_details", None)
                    if details is not None:
                        t.reasoning_tokens = getattr(details, "reasoning_tokens", None)

            elif etype == "error":
                t.error = str(getattr(event, "error", event))

        if t.t_complete is None:
            t.t_complete = time.perf_counter()

    except Exception as e:
        t.error = f"{type(e).__name__}: {e}"
        t.t_complete = time.perf_counter()

    t.finalize()
    return t


def run_anthropic_trial(client: anthropic.Anthropic, model: str, effort: str, trial_idx: int,
                        messages: list[dict]) -> Trial:
    """effort in {off, low, medium, high}."""
    t = Trial(provider="anthropic", model=model, effort=effort, trial_idx=trial_idx)
    system, msgs = anthropic_system_and_messages(messages)

    kwargs: dict[str, Any] = dict(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=msgs,
    )
    if effort == "off":
        # omit thinking parameter entirely
        pass
    else:
        thinking: dict[str, Any] = {"type": "adaptive"}
        # On Opus 4.7 default display is "omitted" (empty text). We need the text
        # to pass to a summarizer, so force "summarized".
        if model == "claude-opus-4-7":
            thinking["display"] = "summarized"
        kwargs["thinking"] = thinking
        kwargs["output_config"] = {"effort": effort}

    t.t_start = time.perf_counter()
    try:
        with client.messages.stream(**kwargs) as stream:
            for event in stream:
                now = time.perf_counter()
                if t.t_first_event is None:
                    t.t_first_event = now

                etype = getattr(event, "type", "")

                if etype == "content_block_start":
                    block = getattr(event, "content_block", None)
                    btype = getattr(block, "type", "")
                    # Track whether current block is thinking to handle stop below.
                    # We'll re-detect at stop by tracking index->type.
                    if not hasattr(t, "_block_types"):
                        t._block_types = {}  # type: ignore[attr-defined]
                    idx = getattr(event, "index", None)
                    if idx is not None:
                        t._block_types[idx] = btype  # type: ignore[attr-defined]

                elif etype == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    dtype = getattr(delta, "type", "")
                    if dtype == "thinking_delta":
                        if t.t_first_reasoning_delta is None:
                            t.t_first_reasoning_delta = now
                        t.reasoning_text += getattr(delta, "thinking", "")
                    elif dtype == "text_delta":
                        if t.t_first_text_delta is None:
                            t.t_first_text_delta = now
                        t.response_text += getattr(delta, "text", "")

                elif etype == "content_block_stop":
                    idx = getattr(event, "index", None)
                    btype = None
                    if hasattr(t, "_block_types") and idx is not None:
                        btype = t._block_types.get(idx)  # type: ignore[attr-defined]
                    if btype == "thinking" and t.t_reasoning_complete is None:
                        t.t_reasoning_complete = now

                elif etype == "message_stop":
                    t.t_complete = now

            # Fetch final message for usage.
            final = stream.get_final_message()
            if final and getattr(final, "usage", None):
                u = final.usage
                t.input_tokens = getattr(u, "input_tokens", None)
                t.output_tokens = getattr(u, "output_tokens", None)
                t.reasoning_tokens = getattr(u, "cache_creation_input_tokens", None) or None

        if t.t_complete is None:
            t.t_complete = time.perf_counter()

    except Exception as e:
        t.error = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        t.error += "\n" + tb[-400:]
        t.t_complete = time.perf_counter()

    # Strip transient attr before serialization
    if hasattr(t, "_block_types"):
        delattr(t, "_block_types")
    t.finalize()
    return t


def run_gemini_trial(client: genai.Client, model: str, effort: str, trial_idx: int,
                     messages: list[dict], include_thoughts: bool = False,
                     max_retries: int = 4) -> Trial:
    """effort in {minimal, low, medium, high}. Maps to ThinkingConfig.thinking_level."""
    label = effort if not include_thoughts else f"{effort}+thoughts"
    system_instruction, contents = gemini_system_and_contents(messages)

    cfg_kwargs: dict[str, Any] = dict(max_output_tokens=MAX_TOKENS)
    if system_instruction:
        cfg_kwargs["system_instruction"] = system_instruction
    cfg_kwargs["thinking_config"] = genai_types.ThinkingConfig(
        thinking_level=effort.upper(),
        include_thoughts=include_thoughts,
    )
    config = genai_types.GenerateContentConfig(**cfg_kwargs)

    attempt = 0
    while True:
        t = Trial(provider="gemini", model=model, effort=label, trial_idx=trial_idx)
        t.t_start = time.perf_counter()
        saw_non_thought = False
        last_thought_time: float | None = None
        try:
            stream = client.models.generate_content_stream(
                model=model, contents=contents, config=config,
            )
            for chunk in stream:
                now = time.perf_counter()
                if t.t_first_event is None:
                    t.t_first_event = now

                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for p in chunk.candidates[0].content.parts:
                        text = getattr(p, "text", None)
                        if not text:
                            continue
                        is_thought = bool(getattr(p, "thought", False))
                        if is_thought:
                            if t.t_first_reasoning_delta is None:
                                t.t_first_reasoning_delta = now
                            t.reasoning_text += text
                            last_thought_time = now
                        else:
                            if not saw_non_thought and last_thought_time is not None:
                                t.t_reasoning_complete = last_thought_time
                            saw_non_thought = True
                            if t.t_first_text_delta is None:
                                t.t_first_text_delta = now
                            t.response_text += text

                um = getattr(chunk, "usage_metadata", None)
                if um is not None:
                    pt = getattr(um, "prompt_token_count", None)
                    ct = getattr(um, "candidates_token_count", None)
                    tt = getattr(um, "thoughts_token_count", None)
                    if pt is not None:
                        t.input_tokens = pt
                    if ct is not None:
                        t.output_tokens = ct
                    if tt is not None:
                        t.reasoning_tokens = tt

            t.t_complete = time.perf_counter()

        except Exception as e:
            t.error = f"{type(e).__name__}: {e}"
            t.t_complete = time.perf_counter()
            # Retry on transient server overload (503 UNAVAILABLE) with backoff.
            err = str(e)
            if attempt < max_retries and ("503" in err or "UNAVAILABLE" in err):
                wait = 2 ** attempt * 5  # 5, 10, 20, 40s
                print(f"    503 retry in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                attempt += 1
                continue
            # Retry 429 only if it's an RPM limit (short retryDelay). Skip daily quotas.
            if attempt < max_retries and "429" in err:
                import re
                m = re.search(r"retry in ([\d.]+)s", err) or re.search(r"retryDelay': '(\d+)s'", err)
                retry_s = float(m.group(1)) if m else 0
                if retry_s and retry_s <= 90:
                    wait = retry_s + 2
                    print(f"    429 RPM retry in {wait:.0f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    attempt += 1
                    continue

        t.finalize()
        return t


def save_trial(trial: Trial, runs_path: Path) -> None:
    with runs_path.open("a") as f:
        f.write(json.dumps(asdict(trial)) + "\n")


def summarize(trial: Trial) -> str:
    def fmt(v):
        return f"{v:.2f}s" if v is not None else "—"
    status = "ERR" if trial.error else "ok "
    return (
        f"  [{status}] {trial.provider:<9} {trial.model:<20} effort={trial.effort:<7} "
        f"trial={trial.trial_idx}  "
        f"first_disp={fmt(trial.first_displayable_s)}  "
        f"reason_done={fmt((trial.t_reasoning_complete - trial.t_start) if trial.t_reasoning_complete else None)}  "
        f"first_text={fmt((trial.t_first_text_delta - trial.t_start) if trial.t_first_text_delta else None)}  "
        f"total={fmt(trial.total_s)}  "
        f"rsn_tok={trial.reasoning_tokens or '—'}  "
        f"rsn_sum_items={trial.reasoning_summary_text_count}"
    )


OPENAI_MODELS = ["gpt-5.2", "gpt-5.4", "gpt-5.4-mini"]
OPENAI_EFFORTS = ["none", "low", "medium", "high"]
ANTHROPIC_MODELS = ["claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6"]
ANTHROPIC_EFFORTS = ["off", "low", "medium", "high"]
# Per-model effort lists for Gemini (different models support different sets).
# Note: gemini-3.1-pro-preview is omitted — requires billing enabled on the GCP
# project; free-tier quota is 0 req/day for this model.
GEMINI_CONFIGS: list[tuple[str, list[str]]] = [
    ("gemma-4-31b-it", ["minimal", "high"]),
    ("gemini-3-flash-preview", ["low", "medium", "high"]),
]


def build_matrix(filter_provider: str | None, filter_model: str | None,
                 filter_effort: str | None, trials: int):
    configs = []
    if filter_provider in (None, "openai"):
        for m in OPENAI_MODELS:
            if filter_model and filter_model != m:
                continue
            for e in OPENAI_EFFORTS:
                if filter_effort and filter_effort != e:
                    continue
                for i in range(trials):
                    configs.append(("openai", m, e, i))
    if filter_provider in (None, "anthropic"):
        for m in ANTHROPIC_MODELS:
            if filter_model and filter_model != m:
                continue
            for e in ANTHROPIC_EFFORTS:
                if filter_effort and filter_effort != e:
                    continue
                for i in range(trials):
                    configs.append(("anthropic", m, e, i))
    if filter_provider in (None, "gemini"):
        for m, efforts in GEMINI_CONFIGS:
            if filter_model and filter_model != m:
                continue
            for e in efforts:
                if filter_effort and filter_effort != e:
                    continue
                for i in range(trials):
                    configs.append(("gemini", m, e, i))
    return configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=Path(__file__).parent,
                    help="Directory containing input.txt input messages; results/ is created there.")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--provider", choices=["openai", "anthropic", "gemini"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--summary", choices=["concise", "detailed", "auto"], default=None,
                    help="OpenAI only: reasoning.summary variant")
    ap.add_argument("--include-thoughts", action="store_true",
                    help="Gemini only: set ThinkingConfig.include_thoughts=True")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    work_dir = Path(args.workdir).resolve()
    results_dir = work_dir / "results"
    results_dir.mkdir(exist_ok=True)
    prompt_path = work_dir / "input.txt"
    runs_path = results_dir / "runs.jsonl"

    load_env()
    prompt = load_prompt(prompt_path)

    configs = build_matrix(args.provider, args.model, args.effort, args.trials)
    print(f"Planning {len(configs)} trials. Output: {runs_path}")
    if args.dry_run:
        for c in configs:
            print(" ", c)
        return

    oai_client = openai.OpenAI()
    ant_client = anthropic.Anthropic()
    gem_client = genai.Client()

    for i, (provider, model, effort, trial_idx) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {provider} {model} effort={effort} trial={trial_idx}")
        if provider == "openai":
            trial = run_openai_trial(oai_client, model, effort, trial_idx, prompt, summary=args.summary)
        elif provider == "anthropic":
            trial = run_anthropic_trial(ant_client, model, effort, trial_idx, prompt)
        else:
            trial = run_gemini_trial(gem_client, model, effort, trial_idx, prompt,
                                     include_thoughts=args.include_thoughts)
        save_trial(trial, runs_path)
        print(summarize(trial))


if __name__ == "__main__":
    main()
