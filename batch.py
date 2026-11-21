#!/usr/bin/env python3
"""Run docupdate over many documents unattended, then print one final report.

  python batch.py INPUT [--out outputs] [--jobs 3] [--limit N] [--resume] [--no-retrieval]

INPUT is either a dataset folder containing cases/<id>/input.json (and optionally score.py), or a
folder of <id>.json files shaped {"sourceDocument": "...", "changes": ["..."]}.

Per document it writes <out>/<id>.md (updated document) or <out>/<id>.fail (reason). A failed
document never stops the batch. Progress is appended to <out>/records.jsonl so --resume continues
after an interruption without re-spending tokens. The final report (also <out>/summary.json) lists
pass/fail counts, token usage, latency percentiles, and how much text was sent to the model.
If the dataset ships a score.py it is run on the outputs and its verdict is printed.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import docupdate as d


def discover(root: str) -> list[tuple[str, str]]:
    cases = os.path.join(root, "cases")
    if os.path.isdir(cases):
        return [(c, os.path.join(cases, c, "input.json")) for c in sorted(os.listdir(cases))
                if os.path.isfile(os.path.join(cases, c, "input.json"))]
    return [(f[:-5], os.path.join(root, f)) for f in sorted(os.listdir(root)) if f.endswith(".json")]


def run_one(cid: str, path: str, out: str, client, retrieval: bool, workers: int) -> dict:
    rec = {"id": cid, "ok": False}
    t0 = time.perf_counter()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        doc, s = d.update_document(data.get("sourceDocument"), data.get("changes"), client,
                                   retrieval=retrieval, workers=workers)
        with open(os.path.join(out, cid + ".md"), "w", encoding="utf-8", newline="") as f:
            f.write(doc)
        rec.update(ok=True, words=s.words_total, words_sent=s.words_sent, windows=s.chunks, mode=s.mode,
                   calls=s.model_calls, retries=s.retries, edits=s.edits_applied,
                   prompt_tokens=s.prompt_tokens, completion_tokens=s.completion_tokens,
                   skipped=[k["reason"] for k in s.skipped])
    except d.UpdateFailed as e:
        rec["error"] = str(e)
    except Exception as e:  # one bad document must never stop the batch
        rec["error"] = f"unexpected {type(e).__name__}"
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    if not rec["ok"]:
        with open(os.path.join(out, cid + ".fail"), "w", encoding="utf-8") as f:
            f.write(rec["error"])
    return rec


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0


def report(records: list[dict], out: str, elapsed: float | None):
    ok = [r for r in records if r["ok"]]
    bad = [r for r in records if not r["ok"]]
    secs = [r["seconds"] for r in records]
    summary = {
        "documents": len(records), "succeeded": len(ok), "failed": len(bad),
        "promptTokens": sum(r.get("prompt_tokens", 0) for r in ok),
        "completionTokens": sum(r.get("completion_tokens", 0) for r in ok),
        "modelCalls": sum(r.get("calls", 0) for r in ok), "retries": sum(r.get("retries", 0) for r in ok),
        "latencySeconds": {"mean": round(statistics.mean(secs), 1) if secs else 0,
                           "p50": pct(secs, .5), "p95": pct(secs, .95), "max": max(secs, default=0)},
        "wordsTotal": sum(r.get("words", 0) for r in ok), "wordsSent": sum(r.get("words_sent", 0) for r in ok),
        "documentsWithSkippedInstructions": sum(1 for r in ok if r.get("skipped")),
        "failureReasons": {},
    }
    for r in bad:
        k = r["error"][:90]
        summary["failureReasons"][k] = summary["failureReasons"].get(k, 0) + 1
    if elapsed is not None:
        summary["wallClockSeconds"] = round(elapsed, 1)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== FINAL REPORT ===")
    print(f"documents: {summary['documents']}   succeeded: {summary['succeeded']}   failed: {summary['failed']}")
    print(f"tokens: prompt {summary['promptTokens']:,}  completion {summary['completionTokens']:,}   "
          f"calls {summary['modelCalls']}  retries {summary['retries']}")
    if summary["wordsTotal"]:
        print(f"text sent to model: {summary['wordsSent']:,} of {summary['wordsTotal']:,} words "
              f"({100 * summary['wordsSent'] / summary['wordsTotal']:.0f}%)")
    L = summary["latencySeconds"]
    print(f"latency per document: mean {L['mean']}s  p50 {L['p50']}s  p95 {L['p95']}s  max {L['max']}s"
          + (f"   wall clock {summary['wallClockSeconds']}s" if elapsed is not None else ""))
    print(f"documents where some instruction was skipped (reported, not applied): "
          f"{summary['documentsWithSkippedInstructions']}")
    for k, v in sorted(summary["failureReasons"].items(), key=lambda kv: -kv[1])[:8]:
        print(f"  failed x{v}: {k}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Run docupdate over many documents and report.")
    ap.add_argument("input")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--jobs", type=int, default=3, help="documents processed in parallel")
    ap.add_argument("--workers", type=int, default=4, help="parallel windows within one document")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume", action="store_true", help="skip documents already recorded in <out>/records.jsonl")
    ap.add_argument("--no-retrieval", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    os.makedirs(args.out, exist_ok=True)
    docs = discover(args.input)[: args.limit]
    rec_path = os.path.join(args.out, "records.jsonl")
    done: dict[str, dict] = {}
    if args.resume and os.path.exists(rec_path):
        with open(rec_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[r["id"]] = r
    elif os.path.exists(rec_path):
        os.remove(rec_path)
    todo = [(c, p) for c, p in docs if c not in done]
    print(f"{len(docs)} documents ({len(done)} already done, {len(todo)} to run)", flush=True)

    try:
        client = d.OpenAIClient()
    except d.UpdateFailed as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    lock = threading.Lock()
    counter = {"n": 0}

    def work(item):
        cid, path = item
        for stale in (".md", ".fail"):
            fp = os.path.join(args.out, cid + stale)
            if os.path.exists(fp):
                os.remove(fp)
        r = run_one(cid, path, args.out, client, not args.no_retrieval, args.workers)
        with lock:
            with open(rec_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            done[cid] = r
            counter["n"] += 1
            tag = "ok  " if r["ok"] else "FAIL"
            print(f"[{counter['n']}/{len(todo)}] {cid} {tag} {r['seconds']}s"
                  + ("" if r["ok"] else f"  {r['error'][:80]}"), flush=True)
        return r

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        list(pool.map(work, todo))
    elapsed = time.perf_counter() - t0
    report([done[c] for c, _ in docs if c in done], args.out, elapsed if todo else None)

    score = os.path.join(args.input, "score.py")
    if os.path.isfile(score) and os.path.isdir(os.path.join(args.input, "cases")):
        print("\n=== DATASET SCORE (score.py) ===", flush=True)
        env = dict(os.environ, PYTHONUTF8="1")
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:   # score only the cases that were run
            os.makedirs(os.path.join(tmp, "cases"))
            shutil.copy(score, tmp)
            for c, _ in docs:
                shutil.copytree(os.path.join(args.input, "cases", c), os.path.join(tmp, "cases", c))
            r = subprocess.run([sys.executable, os.path.join(tmp, "score.py"), tmp, args.out], env=env,
                               capture_output=True, text=True, encoding="utf-8")
        print(r.stdout.rstrip())
        if r.stderr.strip():
            print(r.stderr.strip()[-500:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
