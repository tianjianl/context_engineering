#!/usr/bin/env python3
"""Positional Probe — controlled position-effect experiment.

Leave-one-out design: for each problem, remove it from the list, keep the
remaining problems as fillers in fixed order, and insert the held-out problem
at every position (1..N). Each problem is its own control across positions.

Usage:
    source ~/.bashrc
    python -m context_rot_prelim.positional_probe \
        --model google/gemini-3-flash-preview \
        --input_file /scratch/dkhasha1/tli104/datasets/hmmt_nov_2025/hmmt_nov_2025.jsonl \
        --output_file /scratch/dkhasha1/tli104/outputs/context_rot/probe_gemini-3-flash_hmmt_nov.jsonl
"""

import argparse, asyncio, json, os, random, re, sys, time
from collections import defaultdict
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.data_utils import strip_thinking
from inference.verify_utils import verify_batch


# ── data loading ───────────────────────────────────────────────────────

def load_problems(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "original_problem" in r:
                r["problem_text"] = r["original_problem"]
            else:
                t = r.get("prompt", "")
                if "Problem:" in t:
                    t = t.split("Problem:", 1)[1]
                if t.strip().endswith("Solution:"):
                    t = t.rsplit("Solution:", 1)[0]
                r["problem_text"] = t.strip()
            if "problem_id" not in r:
                r["problem_id"] = f"prob_{r.get('problem_idx', len(rows))}"
            rows.append(r)
    return rows


# ── prompt & parsing ───────────────────────────────────────────────────

PROMPT_TPL = """\
Solve the following {n} math problems in order. For each problem, show your \
reasoning step by step and provide your final answer in \\boxed{{}}.

Separate your solutions using the delimiters shown below.

{block}

Now solve each problem in order. Use "=== Solution N ===" before each \
solution and put your final answer in \\boxed{{}}."""


def build_prompt(questions):
    parts = [f"=== Problem {i} ===\n{q['problem_text']}"
             for i, q in enumerate(questions, 1)]
    return PROMPT_TPL.format(n=len(questions), block="\n\n".join(parts))


def extract_boxed(text):
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth, pos = 1, start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    return text[start:pos - 1].strip() if depth == 0 else None


def parse_sections(response, n):
    cleaned = strip_thinking(response)
    pat = r"===\s*Solution\s+(\d+)\s*==="
    splits = list(re.finditer(pat, cleaned, re.IGNORECASE))
    if len(splits) < 2:
        pat = r"===\s*Problem\s+(\d+)\s*==="
        splits = list(re.finditer(pat, cleaned, re.IGNORECASE))
    out = []
    if len(splits) >= 2:
        secs = {}
        for i, m in enumerate(splits):
            qn = int(m.group(1))
            s = m.end()
            e = splits[i + 1].start() if i + 1 < len(splits) else len(cleaned)
            secs[qn] = cleaned[s:e].strip()
        for q in range(1, n + 1):
            sec = secs.get(q, "")
            out.append((sec, extract_boxed(sec) if sec else None))
    else:
        bpat = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        boxes = re.findall(bpat, cleaned)
        for q in range(n):
            out.append(("", boxes[q].strip() if q < len(boxes) else None))
    return out


# ── verification ───────────────────────────────────────────────────────

def _verify(items):
    if not items:
        return []
    try:
        return verify_batch(items, timeout=10.0)
    except Exception:
        from inference.verify_utils import _verify_single
        return [_verify_single(it) for it in items]


# ── API call ───────────────────────────────────────────────────────────

async def api_call(client, model, msgs, max_tok, temp, retries=8):
    for att in range(retries):
        try:
            r = await client.chat.completions.create(
                model=model, messages=msgs,
                max_tokens=max_tok, temperature=temp)
            c = r.choices[0]
            return {"content": c.message.content or "",
                    "finish_reason": c.finish_reason,
                    "ptok": r.usage.prompt_tokens if r.usage else 0,
                    "ctok": r.usage.completion_tokens if r.usage else 0}
        except Exception as e:
            w = min(2 ** att + random.random(), 60)
            if att < retries - 1:
                print(f"  API err ({att + 1}): {e}, retry {w:.0f}s")
                await asyncio.sleep(w)
            else:
                return {"content": "", "finish_reason": "error",
                        "ptok": 0, "ctok": 0, "error": str(e)}
    return {"content": "", "finish_reason": "error", "ptok": 0, "ctok": 0}


# ── main ───────────────────────────────────────────────────────────────

async def async_main():
    ap = argparse.ArgumentParser(
        description="Positional probe: leave-one-out position effect test")
    ap.add_argument("--model", default="google/gemini-3-flash-preview")
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--max_tokens", type=int, default=65536)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--output_file", required=True)
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

    probs = load_problems(args.input_file)
    N = len(probs)

    print(f"{'=' * 60}\nPositional Probe (leave-one-out)\n{'=' * 60}")
    print(f"Model      : {args.model}")
    print(f"Dataset    : {args.input_file}")
    print(f"Problems   : {N}")
    print(f"Batch size : {N} (leave-one-out)")
    print(f"Total calls: {N * N}")
    print(f"Concurrency: {args.concurrency}")
    print(f"{'=' * 60}")

    # checkpoint
    ckpt = args.output_file + ".ckpt"
    results = []
    done = set()
    if Path(ckpt).exists():
        results = json.load(open(ckpt))
        done = {(r["problem_id"], r["position"]) for r in results}
        print(f"Checkpoint: {len(done)} done")

    sem = asyncio.Semaphore(args.concurrency)
    cnt = [len(done)]
    total = N * N
    t0 = time.time()

    async def probe(held_out_idx, pos):
        q = probs[held_out_idx]
        if (q["problem_id"], pos) in done:
            return

        # Build batch: fillers = all problems except held_out, in original order
        fillers = probs[:held_out_idx] + probs[held_out_idx + 1:]
        batch = fillers[:pos - 1] + [q] + fillers[pos - 1:]
        assert len(batch) == N

        async with sem:
            r = await api_call(
                client, args.model,
                [{"role": "user", "content": build_prompt(batch)}],
                args.max_tokens, args.temperature)

            secs = parse_sections(r["content"], N)
            sec, box = secs[pos - 1] if pos - 1 < len(secs) else ("", None)
            gt = q.get("answer", "")
            vfy = _verify([(gt, sec if sec else r["content"])])
            ok, _, pa = vfy[0] if vfy else (False, "error", None)

            cnt[0] += 1
            el = time.time() - t0
            rpm = cnt[0] / el * 60 if el else 0
            eta = (total - cnt[0]) / rpm if rpm else 0
            m = "+" if ok else "-"
            print(f"  [{cnt[0]}/{total}] {q['problem_id']} @pos{pos}: "
                  f"[{m}] box={box} gt={gt}  ({rpm:.1f}/min, ~{eta:.0f}m)")

            row = {"problem_id": q["problem_id"], "position": pos,
                   "is_correct": ok, "extracted_answer": box,
                   "parsed_answer": pa, "ground_truth": gt,
                   "held_out_idx": held_out_idx,
                   "prompt_tokens": r["ptok"],
                   "completion_tokens": r["ctok"],
                   "finish_reason": r["finish_reason"],
                   "model": args.model}
            results.append(row)
            done.add((q["problem_id"], pos))
            if cnt[0] % 20 == 0:
                json.dump(results, open(ckpt, "w"))

    # Launch all probes
    await asyncio.gather(*(
        probe(qi, p)
        for qi in range(N) for p in range(1, N + 1)))

    json.dump(results, open(ckpt, "w"))

    # Save final
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    results.sort(key=lambda r: (r["problem_id"], r["position"]))
    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    Path(ckpt).unlink(missing_ok=True)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Per-position accuracy (N={N} held-out questions each)")
    print(f"{'=' * 60}")
    by_pos = defaultdict(list)
    for r in results:
        by_pos[r["position"]].append(r["is_correct"])
    for p in sorted(by_pos):
        a = by_pos[p]
        print(f"  Pos {p:2d}: {sum(a)}/{len(a)} = {sum(a) / len(a):.1%}")
    c = sum(r["is_correct"] for r in results)
    print(f"  All   : {c}/{len(results)} = {c / len(results):.1%}")

    # Per-question slope
    print(f"\nPer-question breakdown:")
    by_q = defaultdict(dict)
    for r in results:
        by_q[r["problem_id"]][r["position"]] = r["is_correct"]
    for qid in sorted(by_q):
        pm = by_q[qid]
        marks = "".join("+" if pm.get(p, False) else "-" for p in range(1, N + 1))
        print(f"  {qid}: {marks}  ({sum(pm.values())}/{len(pm)})")

    print(f"\n-> {args.output_file}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
