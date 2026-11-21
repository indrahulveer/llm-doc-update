#!/usr/bin/env python3
"""LLM-driven document updates: the model proposes edits, the program validates and applies them.

Usage:  python docupdate.py input.json [-o output.json]
Needs:  OPENAI_API_KEY in the environment (stdlib only, no pip installs).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

CHUNK_WORDS = 1500        # target words per chunk (paragraph-aligned)
MAX_ATTEMPTS = 3          # model attempts per chunk (1 initial + 2 corrective retries)
MAX_TRANSPORT_RETRIES = 3 # HTTP 429/5xx/network retries per call
MAX_EDITS_PER_CHUNK = 200
WORKERS = 4


class UpdateFailed(Exception):
    """Recovery exhausted or input invalid. No document is ever returned with this."""


class InvalidEdits(Exception):
    """A model response failed validation; message is fed back to the model on retry."""


# --------------------------------------------------------------------------- chunking
def chunk_document(doc: str, max_words: int = CHUNK_WORDS) -> list[str]:
    """Split on blank-line boundaries into chunks of ~max_words. ''.join(chunks) == doc."""
    parts = re.split(r"(\n\s*\n)", doc)  # paragraphs and their separators, kept
    paras = ["".join(parts[i:i + 2]) for i in range(0, len(parts), 2)]
    chunks, cur, cur_words = [], "", 0
    for p in paras:
        w = len(p.split())
        if cur and cur_words + w > max_words:
            chunks.append(cur)
            cur, cur_words = "", 0
        cur += p
        cur_words += w
    if cur:
        chunks.append(cur)
    assert "".join(chunks) == doc
    return chunks


# --------------------------------------------------------------------------- validation
def parse_edits(raw: str) -> list[tuple[str, str]]:
    """Parse the model's JSON into [(find, replace)]; raise InvalidEdits on schema problems."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise InvalidEdits(f"Response was not valid JSON ({e}). Reply with only a JSON object.")
    if not isinstance(data, dict) or not isinstance(data.get("edits"), list):
        raise InvalidEdits('Response must be an object like {"edits": [{"find": "...", "replace": "..."}]}.')
    if len(data["edits"]) > MAX_EDITS_PER_CHUNK:
        raise InvalidEdits(f"Too many edits (max {MAX_EDITS_PER_CHUNK}). Use larger find spans.")
    out = []
    for i, e in enumerate(data["edits"]):
        if (not isinstance(e, dict) or not isinstance(e.get("find"), str)
                or not isinstance(e.get("replace"), str)):
            raise InvalidEdits(f'edits[{i}] must have string fields "find" and "replace".')
        out.append((e["find"], e["replace"]))
    return out


def apply_edits(chunk: str, edits: list[tuple[str, str]]) -> str:
    """Validate every edit against the chunk, then apply all or raise InvalidEdits."""
    spans = []
    for i, (find, replace) in enumerate(edits):
        if not find:
            raise InvalidEdits(f"edits[{i}].find is empty.")
        n = chunk.count(find)
        if n == 0:
            raise InvalidEdits(f"edits[{i}].find does not occur verbatim in the text: {find[:80]!r}. "
                               "Copy it exactly, including punctuation and whitespace.")
        if n > 1:
            raise InvalidEdits(f"edits[{i}].find occurs {n} times: {find[:80]!r}. "
                               "Include more surrounding text so it matches exactly once.")
        if len(replace) > 5 * len(find) + 2000:
            raise InvalidEdits(f"edits[{i}].replace is implausibly large relative to its find.")
        start = chunk.index(find)
        spans.append((start, start + len(find), replace))
    spans.sort()
    for (_, e1, _), (s2, _, _) in zip(spans, spans[1:]):
        if s2 < e1:
            raise InvalidEdits("Two edits overlap. Merge them into one edit.")
    out, pos = [], 0
    for s, e, r in spans:
        out += [chunk[pos:s], r]
        pos = e
    out.append(chunk[pos:])
    return "".join(out)


# --------------------------------------------------------------------------- model client
class OpenAIClient:
    """Minimal Chat Completions client (stdlib). Reads the key from OPENAI_API_KEY only."""

    def __init__(self, model: str | None = None, timeout: float = 120):
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise UpdateFailed("OPENAI_API_KEY is not set.")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.timeout = timeout

    def complete(self, messages: list[dict]) -> tuple[str, dict]:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0,
                           "response_format": {"type": "json_object"}}).encode()
        for attempt in range(MAX_TRANSPORT_RETRIES + 1):
            req = urllib.request.Request(
                f"{self.base}/chat/completions", data=body, method="POST",
                headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.load(r)
                choice = data["choices"][0]
                text = choice["message"].get("content") or ""
                if choice.get("finish_reason") == "length":
                    text = ""  # truncated output is never trusted; fails JSON parse and is retried
                return text, data.get("usage", {})
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < MAX_TRANSPORT_RETRIES:
                    time.sleep(2 ** attempt)
                    continue
                raise UpdateFailed(f"OpenAI API HTTP {e.code}")  # body deliberately not logged
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as e:
                if attempt < MAX_TRANSPORT_RETRIES:
                    time.sleep(2 ** attempt)
                    continue
                raise UpdateFailed(f"OpenAI API call failed: {type(e).__name__}")
        raise UpdateFailed("unreachable")


SYSTEM = """You edit one part of a larger document by proposing exact text replacements.
Reply with ONLY a JSON object: {"edits": [{"find": "<exact text>", "replace": "<new text>"}]}
Rules:
- "find" must be copied verbatim from the text and occur exactly once in it; add surrounding words if needed.
- Edits must not overlap. Keep each find as small as possible so unrelated text is untouched.
- To delete a paragraph, set "replace" to "" and include the paragraph and one adjacent blank line in "find".
- For document-wide changes (e.g. tone), edit every affected sentence or paragraph.
- Apply every requested change that is relevant to this part. If nothing here is affected, return {"edits": []}.
- Never invent facts. Change only what the instructions require."""


def build_messages(chunk: str, changes: list[str]) -> list[dict]:
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(changes))
    user = (f"Requested changes (apply to this part of the document):\n{numbered}\n\n"
            f"--- DOCUMENT PART START ---\n{chunk}\n--- DOCUMENT PART END ---")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- orchestration
@dataclass
class Stats:
    chunks: int = 0
    model_calls: int = 0
    retries: int = 0
    edits_applied: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    log: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def update_chunk(client, chunk: str, changes: list[str], idx: int, stats: Stats) -> tuple[str, int]:
    messages = build_messages(chunk, changes)
    last_err = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw, usage = client.complete(messages)
        with stats.lock:
            stats.model_calls += 1
            stats.prompt_tokens += usage.get("prompt_tokens", 0)
            stats.completion_tokens += usage.get("completion_tokens", 0)
        try:
            edits = parse_edits(raw)
            return apply_edits(chunk, edits), len(edits)
        except InvalidEdits as e:
            last_err = str(e)
            with stats.lock:
                stats.retries += 1
                stats.log.append(f"chunk {idx} attempt {attempt}: {last_err}")  # find snippets only
            messages = messages + [
                {"role": "assistant", "content": raw[:2000]},
                {"role": "user", "content": f"Your response was rejected: {last_err}\n"
                                            "Return a corrected JSON object for the same task."}]
    raise UpdateFailed(f"chunk {idx} failed after {MAX_ATTEMPTS} attempts; last error: {last_err}")


def update_document(source: str, changes: list[str], client, chunk_words: int = CHUNK_WORDS,
                    workers: int = WORKERS) -> tuple[str, Stats]:
    """All-or-nothing: returns the full updated document or raises UpdateFailed."""
    if not isinstance(source, str) or not source.strip():
        raise UpdateFailed("sourceDocument must be a non-empty string.")
    if (not isinstance(changes, list) or not changes
            or not all(isinstance(c, str) and c.strip() for c in changes)):
        raise UpdateFailed("changes must be a non-empty list of non-empty strings.")
    chunks = chunk_document(source, chunk_words)
    stats = Stats(chunks=len(chunks))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(update_chunk, client, c, changes, i, stats) for i, c in enumerate(chunks)]
        results = []
        try:
            for f in futures:
                results.append(f.result())  # first failure aborts the whole job
        except BaseException:
            for f in futures:
                f.cancel()
            raise
    stats.edits_applied = sum(n for _, n in results)
    result = "".join(t for t, _ in results)
    if result == source:
        raise UpdateFailed("model proposed no effective changes; refusing to report success.")
    return result, stats


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Apply natural-language changes to a document via OpenAI.")
    ap.add_argument("input", help='JSON file: {"sourceDocument": "...", "changes": ["..."]}')
    ap.add_argument("-o", "--output", help="write result JSON here (default: stdout)")
    args = ap.parse_args(argv)

    def emit(obj, code):
        text = json.dumps(obj, indent=2, ensure_ascii=False)
        if args.output:
            tmp = args.output + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, args.output)
        else:
            print(text)
        return code

    try:
        with open(args.input, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise UpdateFailed("input must be a JSON object.")
        doc, stats = update_document(data.get("sourceDocument"), data.get("changes"), OpenAIClient())
    except (OSError, json.JSONDecodeError) as e:
        return emit({"status": "error", "error": f"cannot read input: {type(e).__name__}"}, 1)
    except UpdateFailed as e:
        return emit({"status": "error", "error": str(e)}, 1)
    s = stats
    return emit({"status": "ok", "updatedDocument": doc,
                 "stats": {"chunks": s.chunks, "modelCalls": s.model_calls, "retries": s.retries,
                           "editsApplied": s.edits_applied, "promptTokens": s.prompt_tokens,
                           "completionTokens": s.completion_tokens}}, 0)


if __name__ == "__main__":
    sys.exit(main())
