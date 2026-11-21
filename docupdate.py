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
from typing import NamedTuple

import selector

CHUNK_WORDS = 1500        # target words per chunk (paragraph-aligned)
MAX_ATTEMPTS = 3          # model attempts per chunk (1 initial + 2 corrective retries)
MAX_TRANSPORT_RETRIES = 5 # HTTP 429/5xx/network retries per call
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
class Edit(NamedTuple):
    find: str
    replace: str
    change: int | None = None   # 1-based number of the instruction this edit serves
    all: bool = False           # replace every occurrence of `find` in the window


def parse_edits(raw: str, valid_changes: set[int] | None = None) -> list[Edit]:
    """Parse the model's JSON into Edits; raise InvalidEdits on schema problems."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise InvalidEdits(f"Response was not valid JSON ({e}). Reply with only a JSON object.")
    if not isinstance(data, dict) or not isinstance(data.get("edits"), list):
        raise InvalidEdits('Response must be an object like {"edits": [{"change": 1, "find": "...", "replace": "..."}]}.')
    if len(data["edits"]) > MAX_EDITS_PER_CHUNK:
        raise InvalidEdits(f"Too many edits (max {MAX_EDITS_PER_CHUNK}). Use larger find spans, or all=true for repeated text.")
    out = []
    for i, e in enumerate(data["edits"]):
        if (not isinstance(e, dict) or not isinstance(e.get("find"), str)
                or not isinstance(e.get("replace"), str)):
            raise InvalidEdits(f'edits[{i}] must have string fields "find" and "replace".')
        ch = e.get("change")
        if ch is not None and (isinstance(ch, bool) or not isinstance(ch, int)
                               or (valid_changes is not None and ch not in valid_changes)):
            raise InvalidEdits(f'edits[{i}].change must be one of the listed instruction numbers '
                               f'{sorted(valid_changes) if valid_changes else ""}.')
        if not isinstance(e.get("all", False), bool):
            raise InvalidEdits(f'edits[{i}].all must be true or false.')
        out.append(Edit(e["find"], e["replace"], ch, bool(e.get("all", False))))
    return out


def apply_edits(chunk: str, edits: list, mark: bool = False) -> str:
    """Validate every edit against the chunk, then apply all or raise InvalidEdits.

    Edits are Edit tuples or plain (find, replace) pairs. mark=True returns a review copy where
    each change is shown as ~~old~~**new**."""
    spans = []
    for i, ed in enumerate(edits):
        find, replace = ed[0], ed[1]
        every = len(ed) > 3 and bool(ed[3])
        if not find:
            raise InvalidEdits(f"edits[{i}].find is empty.")
        n = chunk.count(find)
        if n == 0:
            raise InvalidEdits(f"edits[{i}].find does not occur verbatim in the text: {find[:80]!r}. "
                               "Copy it exactly, including punctuation and whitespace.")
        if n > 1 and not every:
            raise InvalidEdits(f"edits[{i}].find occurs {n} times: {find[:80]!r}. "
                               "Include more surrounding text so it matches exactly once, "
                               'or set "all": true if every occurrence must change.')
        if len(replace) > 5 * len(find) + 2000:
            raise InvalidEdits(f"edits[{i}].replace is implausibly large relative to its find.")
        k = chunk.find(find)
        if not every and k > 0 and find[0] in "#*" and chunk[k - 1] == find[0]:
            raise InvalidEdits(f"edits[{i}].find starts in the middle of a markdown marker "
                               f"({chunk[max(0, k - 3):k + 3]!r}). Start at the beginning of the line.")
        pos = 0
        while True:
            start = chunk.find(find, pos)
            if start < 0:
                break
            spans.append((start, start + len(find), replace))
            pos = start + len(find)
            if not every:
                break
    spans.sort()
    for (_, e1, _), (s2, _, _) in zip(spans, spans[1:]):
        if s2 < e1:
            raise InvalidEdits("Two edits overlap. Merge them into one edit.")
    out, pos = [], 0
    for s_, e, r in spans:
        out.append(chunk[pos:s_])
        if mark:
            out.append(_highlight(chunk[s_:e], r))
        else:
            out.append(r)
            if not r.strip():      # a deletion must not leave a run of blank lines behind
                left = "".join(out)
                have = len(left) - len(left.rstrip(chr(10)))
                rest = chunk[e:]
                lead = len(rest) - len(rest.lstrip(chr(10)))
                e += max(0, min(lead, have + lead - 2))
        pos = e
    out.append(chunk[pos:])
    return "".join(out)


def _highlight(old: str, new: str) -> str:
    """Markdown review markup: ~~removed~~ **added** (surrounding whitespace kept outside the markers)."""
    def wrap(text, m):  # mark each line separately so Markdown renders multi-paragraph spans
        def one(line):
            core = line.strip()
            if not core:
                return line
            i = line.index(core)
            return line[:i] + m + core + m + line[i + len(core):]
        return chr(10).join(one(l) for l in text.split(chr(10)))
    return wrap(old, "~~") + wrap(new, "**")


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
                    try:
                        wait = float(e.headers.get("Retry-After", ""))
                    except (TypeError, ValueError):
                        wait = 0.0
                    time.sleep(min(60.0, max(wait, 2 ** attempt)))
                    continue
                raise UpdateFailed(f"OpenAI API HTTP {e.code}")  # body deliberately not logged
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as e:
                if attempt < MAX_TRANSPORT_RETRIES:
                    time.sleep(2 ** attempt)
                    continue
                raise UpdateFailed(f"OpenAI API call failed: {type(e).__name__}")
        raise UpdateFailed("unreachable")


SYSTEM = """You edit one part of a larger document by proposing exact text replacements.
Reply with ONLY a JSON object:
{"edits": [{"change": <instruction number>, "find": "<exact text>", "replace": "<new text>", "all": false}]}
Rules:
- "change" is the number of the requested change the edit serves.
- "find" must be copied verbatim from the text. Unless "all" is true it must occur exactly once in it; add surrounding words if needed.
- Set "all": true only for a literal substitution that must hit every occurrence in this part (e.g. renaming a product):
  then "find" is the short literal and every occurrence is replaced. Renames must also cover derived forms of the
  name (e.g. lowercase or joined in URLs and code: "Nimbus Desk" -> "nimbusdesk") with their own edit.
- Edits must not overlap. Keep each find as small as possible so unrelated text is untouched.
- To delete a section or paragraph, set "replace" to "" and include its heading/paragraph and one adjacent blank line in "find".
- Apply price or number changes exactly (e.g. +10% of $50 is $55) and only to the items the change names;
  monthly and annual lists are separate: never edit the one the change does not name.
- If a requested change refers to something that does not exist in this part, make no edit for it. Never guess a different target.
- For document-wide changes (e.g. tone), edit every affected sentence or paragraph, keeping every fact, number, link, heading and code block unchanged.
- If nothing here is affected, return {"edits": []}. Never invent facts. Change only what the instructions require."""


def build_messages(chunk: str, changes: list) -> list[dict]:
    """changes: [(instruction number, text)] or plain strings (numbered from 1)."""
    pairs = [c if isinstance(c, tuple) else (i + 1, c) for i, c in enumerate(changes)]
    nl = chr(10)
    numbered = nl.join(f"{n}. {t}" for n, t in pairs)
    user = (f"Requested changes (apply to this part of the document):{nl}{numbered}{nl}{nl}"
            f"--- DOCUMENT PART START ---{nl}{chunk}{nl}--- DOCUMENT PART END ---")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- orchestration
@dataclass
class Stats:
    chunks: int = 0           # model windows sent
    mode: str = ""
    words_total: int = 0
    words_sent: int = 0
    model_calls: int = 0
    retries: int = 0
    edits_applied: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    log: list = field(default_factory=list)
    changes: dict = field(default_factory=dict)   # window idx -> [{before, after}]
    applied_by: dict = field(default_factory=dict) # instruction number -> edits attributed to it
    rewrite_left_unchanged: int = 0
    skipped: list = field(default_factory=list)   # [{instruction, text, reason}] not sent / not applied
    highlighted: str = ""                         # full document with changes marked up
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def update_chunk(client, chunk: str, changes: list, idx: int, stats: Stats) -> tuple[str, int, str]:
    messages = build_messages(chunk, changes)
    valid = {c[0] for c in changes if isinstance(c, tuple)} or None
    last_err = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw, usage = client.complete(messages)
        with stats.lock:
            stats.model_calls += 1
            stats.prompt_tokens += usage.get("prompt_tokens", 0)
            stats.completion_tokens += usage.get("completion_tokens", 0)
        try:
            edits = parse_edits(raw, valid)
            check_destructive(edits, [c for c in changes if isinstance(c, tuple)])
            new = apply_edits(chunk, edits)
            with stats.lock:
                stats.changes[idx] = [{"before": e.find, "after": e.replace} for e in edits if e.find != e.replace]
                for e in edits:
                    if e.change is not None and e.find != e.replace:
                        stats.applied_by[e.change] = stats.applied_by.get(e.change, 0) + 1
            return new, len(edits), apply_edits(chunk, edits, mark=True)
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


_QUOTED = re.compile(r"""(?<!\w)(['"‘“])(.{2,80}?)['"’”](?!\w)""")
_REPLACING = re.compile(r"\b(replace|change|rename|substitute|swap)\b", re.I)
_EVERY = re.compile(r"\b(all|every|everywhere|each|any)\b", re.I)


def check_anchor(source: str, instruction: str) -> str | None:
    """Deterministic grounding of quoted targets. Returns a reason to skip the instruction, else None.

    A quoted string is a literal anchor: it must exist, and for replace-style instructions the
    first quoted string (the thing being replaced) must be unambiguous unless the instruction
    says all/every/everywhere. Unquoted targets are left to the model."""
    m = _QUOTED.search(instruction)
    if not m:
        return _check_named_entities(source, instruction)
    target = m.group(2)
    n = len(re.findall(re.escape(target), source, flags=re.I))
    if n == 0:
        return f"target {target!r} not found in the document"
    if n > 1 and _REPLACING.search(instruction) and not _EVERY.search(instruction):
        return f"ambiguous: {target!r} occurs {n} times, so the intended occurrence is unclear"
    return None


_EDIT_VERB = re.compile(r"\s*(change|update|remove|delete|raise|increase|decrease|lower|rename|replace|set|fix)\b", re.I)
_SPLIT = re.compile(r"\s(?:to|with|into|by)\s", re.I)
_PROPER = re.compile(r"\b[A-Z][a-z]{2,}\b")


def _check_named_entities(source: str, instruction: str) -> str | None:
    """A capitalised name on the *target side* of an edit instruction ("Change the Platinum tier price
    to $999") must occur in the document, otherwise the target does not exist and the model would
    only guess. New names after "to/with/into" are not checked. Edit verbs only (not add/insert)."""
    if _EDIT_VERB.match(instruction) is None:
        return None
    target_side = _SPLIT.split(instruction, maxsplit=1)[0]
    target_side = re.split(r"\sfrom\s", target_side, maxsplit=1, flags=re.I)[0]
    words = _PROPER.findall(target_side)[1:]          # [0] is the verb at the start
    for w in words:
        if not re.search(re.escape(w), source, flags=re.I):
            return f"refers to {w!r}, which does not appear in the document"
    if re.match(r"\s*(update|change|set)\b", instruction, re.I):   # "set X to Y": X must exist
        have = set(selector.tokenize(source))
        for w in selector.tokenize(target_side):
            if len(w) >= 4 and not w.isdigit() and w not in have:
                return f"its target mentions {w!r}, which does not appear in the document"
    return None


def check_destructive(edits: list, pairs: list) -> None:
    """Large deletions must share at least one content word with the instruction they claim to serve."""
    by_n = dict(pairs)
    every = set(t for _, text in pairs for t in selector.tokenize(text))
    for i, e in enumerate(edits):
        if len(e.find) >= 60 and len(e.replace) < 0.5 * len(e.find):
            terms = set(selector.tokenize(by_n[e.change])) if e.change in by_n else every
            if not terms & set(selector.tokenize(e.find)):
                raise InvalidEdits(
                    f"edits[{i}] deletes text that shares no words with the instruction it serves. "
                    "Only remove text the instruction names; if it is not in this part, return no edit.")


_RENAME = re.compile(
    r"^\s*rename\b[^.]*?\bfrom\s+(?P<a>.+?)\s+to\s+(?P<b>.+?)"
    r"(?:\s+(?:everywhere|throughout|in|across|wherever)\b.*?)?[.!]?\s*$", re.I)


def literal_rename(source: str, instruction: str):
    """Deterministic path for "Rename X to Y [everywhere]": a literal substitution needs no LLM.

    Also renames derived forms of the name (joined/hyphenated/underscored lowercase, as in URLs and code).
    Returns None if the instruction is not a literal rename, else (new_source, count) or an error string."""
    m = _RENAME.match(instruction)
    if not m:
        return None
    strip = lambda x: x.strip().strip("'\"\u2018\u2019\u201c\u201d")
    a, b = strip(m.group("a")), strip(m.group("b"))
    if not a or not b or a == b:
        return None
    if a not in source:
        return f"target {a!r} not found in the document"
    forms = [(a, b)]
    for sep in ("", "-", "_"):
        fa, fb = sep.join(a.lower().split()), sep.join(b.lower().split())
        if fa != a and (fa, fb) not in forms:
            forms.append((fa, fb))
    total = 0
    for fa, fb in forms:
        total += source.count(fa)
        source = source.replace(fa, fb)
    return source, total


REWRITE_WORDS = 350      # words per call in the paragraph-rewrite stage
_PROTECT = re.compile(r"https?://[^\s)>\]]+|`[^`\n]+`|\*\*[^*\n]+\*\*|[$\u20ac\u00a3]?\d[\d,]*(?:\.\d+)?%?")
_LEAD = re.compile(r"^\s*(?:[-*+]|\d+[.)]|>)\s+")

SYSTEM_REWRITE = """You rewrite paragraphs of a document to satisfy the requested style changes.
Paragraphs are given as [id] text. Reply with ONLY a JSON object:
{"paragraphs": [{"id": <id>, "text": "<rewritten paragraph>"}]}
Rules:
- Include only paragraphs you actually change. Rewrite each as ONE paragraph, same language.
- Keep every fact unchanged: all numbers, prices, percentages, product and plan names, URLs, code and
  `inline code`, and **bold** terms must appear exactly as in the original.
- Keep list markers ("- ", "1. ", "> ") exactly as they are. Never add or drop information.
- If a paragraph already satisfies the request, leave it out."""


def _rewritable(paras: list[str]) -> list[bool]:
    """Prose paragraphs only: not headings, not inside code fences."""
    flags, in_fence = [], False
    for p in paras:
        fence = p.count("```")
        flags.append(bool(p.strip()) and not in_fence and fence == 0 and not p.lstrip().startswith("#"))
        if fence % 2:
            in_fence = not in_fence
    return flags


def check_rewrite(orig: dict, raw: str) -> tuple[dict, dict]:
    """Validate a paragraph-rewrite response. Returns (accepted {id: text}, rejected {id: reason}).

    A malformed response as a whole raises InvalidEdits; individual bad paragraphs are rejected one by one."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise InvalidEdits(f"Response was not valid JSON ({e}). Reply with only a JSON object.")
    if not isinstance(data, dict) or not isinstance(data.get("paragraphs"), list):
        raise InvalidEdits('Response must be {"paragraphs": [{"id": 1, "text": "..."}]}.')
    accepted, rejected = {}, {}
    for item in data["paragraphs"]:
        if (not isinstance(item, dict) or isinstance(item.get("id"), bool) or not isinstance(item.get("id"), int)
                or not isinstance(item.get("text"), str)):
            raise InvalidEdits('Each paragraph needs an integer "id" and a string "text".')
        i, text = item["id"], item["text"].strip()
        if i not in orig:
            raise InvalidEdits(f"id {i} is not one of the paragraphs given ({min(orig)}-{max(orig)}).")
        if i in accepted or i in rejected:
            raise InvalidEdits(f"id {i} appears twice.")
        old = orig[i].strip()
        nl2 = chr(10) * 2
        why = None
        if not text or (nl2 in text and nl2 not in old):
            why = "must be a single non-empty paragraph"
        elif not (0.4 * len(old) <= len(text) <= 3 * len(old) + 40):
            why = "changed length implausibly; rewrite it, do not drop or pad content"
        else:
            lm_old, lm_new = _LEAD.match(old), _LEAD.match(text)
            if (lm_old.group(0).strip() if lm_old else "") != (lm_new.group(0).strip() if lm_new else ""):
                why = "must keep its list/quote marker exactly"
            elif any(l.lstrip().startswith(("#", "```")) for l in text.split(chr(10))) and not any(
                    l.lstrip().startswith(("#", "```")) for l in old.split(chr(10))):
                why = "must not introduce headings or code fences"
            else:
                missing = [t for t in dict.fromkeys(_PROTECT.findall(old)) if t not in text]
                if missing:
                    why = f"lost protected text that must stay verbatim: {missing[:5]}"
        if why:
            rejected[i] = why
        else:
            accepted[i] = text
    return accepted, rejected


def rewrite_window(client, items: list, pairs: list, key: int, stats: Stats) -> dict:
    """items: [(id, paragraph)]. Returns {id: new_text} for paragraphs that changed.

    Paragraphs that still fail validation after the retries are left unchanged (never half-edited)
    and counted in stats.rewrite_left_unchanged, which update_document reports as skipped."""
    nl = chr(10)
    numbered = nl.join(f"{n}. {t}" for n, t in pairs)
    body = (nl * 2).join(f"[{i}] {p.strip()}" for i, p in items)
    messages = [{"role": "system", "content": SYSTEM_REWRITE},
                {"role": "user", "content": f"Requested changes:{nl}{numbered}{nl}{nl}{body}"}]
    orig = dict(items)
    good: dict = {}
    pending = set(orig)
    last = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw, usage = client.complete(messages)
        with stats.lock:
            stats.model_calls += 1
            stats.prompt_tokens += usage.get("prompt_tokens", 0)
            stats.completion_tokens += usage.get("completion_tokens", 0)
        try:
            accepted, rejected = check_rewrite({i: orig[i] for i in pending}, raw)
        except InvalidEdits as e:
            last = str(e)
            rejected = None
        if rejected is not None:
            good.update(accepted)
            pending = set(rejected)      # omitted paragraphs count as "unchanged"; only rejected ones are re-asked
            if not rejected:
                break
            last = "; ".join(f"[{i}] {why}" for i, why in sorted(rejected.items()))
            if attempt == MAX_ATTEMPTS:
                break
            reask = {i: orig[i] for i in sorted(pending)}
            messages = [{"role": "system", "content": SYSTEM_REWRITE},
                        {"role": "user", "content": f"Requested changes:{nl}{numbered}{nl}{nl}"
                         + (nl * 2).join(f"[{i}] {p.strip()}" for i, p in reask.items())},
                        {"role": "assistant", "content": raw[:2000]},
                        {"role": "user", "content": f"Some paragraphs were rejected: {last}{nl}"
                                                    "Return corrected JSON for ONLY those paragraphs "
                                                    "(or omit a paragraph to leave it unchanged)."}]
        else:
            if attempt == MAX_ATTEMPTS:
                raise UpdateFailed(f"rewrite window {key} failed after {MAX_ATTEMPTS} attempts; last error: {last}")
            messages = messages + [
                {"role": "assistant", "content": raw[:2000]},
                {"role": "user", "content": f"Your response was rejected: {last}{nl}"
                                            "Return a corrected JSON object for the same task."}]
        with stats.lock:
            stats.retries += 1
            stats.log.append(f"rewrite window {key} attempt {attempt}: {last[:200]}")
    left = sorted(pending - set(good))
    new = {i: t for i, t in good.items() if t != orig[i].strip()}
    with stats.lock:
        stats.changes[key] = [{"before": orig[i].strip(), "after": t} for i, t in new.items()]
        if left:
            stats.rewrite_left_unchanged += len(left)
    return new


def rewrite_document(text: str, pairs: list, client, stats: Stats, workers: int) -> str:
    """Document-wide stage: paragraph-addressed rewrite (no verbatim anchors to get wrong)."""
    parts = re.split(r"(\n\s*\n)", text)
    idx = list(range(0, len(parts), 2))
    ok = _rewritable([parts[i] for i in idx])
    windows, cur, words = [], [], 0
    for n, i in enumerate(idx):
        if not ok[n]:
            continue
        w = len(parts[i].split())
        if cur and words + w > REWRITE_WORDS:
            windows.append(cur)
            cur, words = [], 0
        cur.append((i, parts[i]))
        words += w
    if cur:
        windows.append(cur)
    stats.chunks += len(windows)
    stats.words_sent += sum(len(p.split()) for win in windows for _, p in win)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(rewrite_window, client, win, pairs, 1_000_000 + k, stats)
                for k, win in enumerate(windows)]
        try:
            results = [f.result() for f in futs]
        except BaseException:
            for f in futs:
                f.cancel()
            raise
    for r in results:
        for i, t in r.items():
            lead = parts[i][: len(parts[i]) - len(parts[i].lstrip())]
            trail = parts[i][len(parts[i].rstrip()):]
            parts[i] = lead + t + trail
            stats.edits_applied += 1
    return "".join(parts)


def diff_highlight(old: str, new: str) -> str:
    """Paragraph-level review copy (used after the rewrite stage): ~~old~~ **new**."""
    import difflib
    a, b = re.split(r"\n\s*\n", old), re.split(r"\n\s*\n", new)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            out += a[i1:i2]
        else:
            out.append(_highlight("".join(a[i1:i2]), "".join(b[j1:j2])))
    return "\n\n".join(x.strip(chr(10)) for x in out)


def run_targeted(text: str, pairs: list, client, stats: Stats, unit_words: int, window_words: int,
                 workers: int, retrieval: bool) -> tuple[str, str]:
    """Targeted stage: BM25-selected windows, find/replace edits. Returns (new_text, highlighted)."""
    act = [t for _, t in pairs]
    if retrieval:
        units = chunk_document(text, unit_words)
        wanted, mode = selector.select(units, act)
    else:
        units = chunk_document(text, window_words)
        wanted, mode = [set(range(len(act))) for _ in units], "full"
    stats.mode = mode
    segs = selector.plan(units, wanted, window_words)
    jobs = [(i, t, [pairs[j] for j in sorted(w)]) for i, (t, w) in enumerate(segs) if w]
    stats.chunks += len(jobs)
    stats.words_sent += sum(len(t.split()) for _, t, _ in jobs)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {i: pool.submit(update_chunk, client, t, ch, i, stats) for i, t, ch in jobs}
        try:
            done = {i: f.result() for i, f in futures.items()}  # first failure aborts the whole job
        except BaseException:
            for f in futures.values():
                f.cancel()
            raise
    stats.edits_applied += sum(v[1] for v in done.values())
    new = "".join(done[i][0] if i in done else t for i, (t, _) in enumerate(segs))
    marked = "".join(done[i][2] if i in done else t for i, (t, _) in enumerate(segs))
    return new, marked


from decimal import Decimal, ROUND_HALF_UP

_PCT = re.compile(r"^\s*(raise|increase|lower|decrease|reduce|cut)\b.*?\bprices?\b.*?\bby\s+(\d+(?:\.\d+)?)\s*(?:%|percent)",
                  re.I | re.S)
_MONEY = re.compile(r"\$(\d[\d,]*(?:\.\d+)?)(\s+per\s+(month|year))", re.I)


def percent_price_change(source: str, instruction: str):
    """Deterministic path for "Raise/lower <scope> prices by N%": exact arithmetic, no LLM.

    Only prices written like "$40 per month" / "$400 per year" are touched. Scope words narrow it:
    monthly or annual (the clause before any ';' or 'leave'), and tier names that appear in the
    document as "**Name**:". Returns None when it does not apply (then the LLM path is used),
    else (new_source, [{before, after}])."""
    m = _PCT.match(instruction)
    if not m:
        return None
    clause = re.split(r";|\bleave\b|\bexcept\b", instruction, maxsplit=1, flags=re.I)[0].lower()
    month = "month" in clause
    year = "annual" in clause or "year" in clause
    periods = {"month"} if month and not year else {"year"} if year and not month else {"month", "year"}
    names = {t.strip() for t in re.findall(r"\*\*([^*\n]{2,30})\*\*\s*:", source)}
    named = [t for t in names if re.search(r"\b" + re.escape(t.lower()) + r"\b", clause)]
    factor = Decimal(1) + (Decimal(m.group(2)) / 100) * (1 if m.group(1).lower() in ("raise", "increase") else -1)
    changes = []

    def bump(mm):
        old, period = mm.group(1), mm.group(3).lower()
        if period not in periods:
            return mm.group(0)
        val = Decimal(old.replace(",", "")) * factor
        val = val.quantize(Decimal("1") if val == val.to_integral_value() else Decimal("0.01"), ROUND_HALF_UP)
        txt = f"{val:,}" if "," in old else f"{val}"
        new = f"${txt}{mm.group(2)}"
        changes.append({"before": mm.group(0), "after": new})
        return new

    out = []
    for line in source.split(chr(10)):
        if named and not any(f"**{t}**" in line for t in named):
            out.append(line)
        else:
            out.append(_MONEY.sub(bump, line))
    if not changes:
        return None
    return chr(10).join(out), changes


def update_document(source: str, changes: list[str], client, unit_words: int = selector.UNIT_WORDS,
                    window_words: int = selector.WINDOW_WORDS, workers: int = WORKERS,
                    retrieval: bool = True) -> tuple[str, Stats]:
    """All-or-nothing: returns the full updated document or raises UpdateFailed.

    Pipeline: (0) literal renames, no LLM  ->  (1) pre-flight grounding of each instruction  ->
    (2) targeted instructions: BM25-selected windows, validated find/replace edits  ->
    (3) document-wide instructions (tone/style): paragraph-addressed rewrite of all prose, with
    protected text (numbers, URLs, code, bold terms, list markers) checked programmatically.
    retrieval=False sends every window to stage 2 instead of only the BM25-selected ones.
    """
    if not isinstance(source, str) or not source.strip():
        raise UpdateFailed("sourceDocument must be a non-empty string.")
    if (not isinstance(changes, list) or not changes
            or not all(isinstance(c, str) and c.strip() for c in changes)):
        raise UpdateFailed("changes must be a non-empty list of non-empty strings.")
    stats = Stats(words_total=len(source.split()))
    original = source
    handled: set[int] = set()
    renames = []
    for i, c in enumerate(changes):
        r = literal_rename(source, c)
        if r is None:
            continue
        handled.add(i)
        if isinstance(r, str):
            stats.skipped.append({"instruction": i + 1, "text": c, "reason": r})
        else:
            source, n = r
            stats.applied_by[i + 1] = n
            m = _RENAME.match(c)
            renames.append({"before": m.group("a").strip(), "after": m.group("b").strip(), "occurrences": n})
    if renames:
        stats.changes[-1] = renames
    prices = []
    for i, c in enumerate(changes):
        if i in handled:
            continue
        r = percent_price_change(source, c)
        if r is not None:
            source, ch = r
            handled.add(i)
            stats.applied_by[i + 1] = len(ch)
            prices += ch
    if prices:
        stats.changes[-2] = prices
    targeted, wide = [], []
    for i, c in enumerate(changes):
        if i in handled:
            continue
        reason = check_anchor(source, c)
        if reason:
            stats.skipped.append({"instruction": i + 1, "text": c, "reason": reason})
        else:
            (wide if selector.GLOBAL_RE.search(c) else targeted).append((i + 1, c))
    if not targeted and not wide and not renames and not prices:
        raise UpdateFailed("no instruction could be applied: " + "; ".join(
            f"#{k['instruction']} {k['reason']}" for k in stats.skipped))

    text, marked = source, source
    stats.mode = "deterministic" if not (targeted or wide) else ""
    if targeted:
        text, marked = run_targeted(text, targeted, client, stats, unit_words, window_words, workers, retrieval)
    if wide:
        stats.mode = (stats.mode + "+" if stats.mode else "") + "rewrite"
        before = text
        text = rewrite_document(text, wide, client, stats, workers)
        if stats.rewrite_left_unchanged:
            stats.skipped.append({"instruction": wide[0][0], "text": wide[0][1],
                                  "reason": f"{stats.rewrite_left_unchanged} paragraph(s) left unchanged: "
                                            "the rewrite failed validation after retries"})
        if text != before:
            for n, _ in wide:
                stats.applied_by[n] = 1
        marked = diff_highlight(original, text)
    stats.highlighted = marked
    if text == original:
        raise UpdateFailed("no effective changes were produced; refusing to report success.")
    if stats.applied_by:   # attribution available: report instructions that changed nothing
        for n, c in targeted + wide:
            if stats.applied_by.get(n, 0) == 0:
                stats.skipped.append({"instruction": n, "text": c,
                                      "reason": "no edit was proposed for it (target not found?)"})
    return text, stats


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Apply natural-language changes to a document via OpenAI.")
    ap.add_argument("input", help='JSON file: {"sourceDocument": "...", "changes": ["..."]}')
    ap.add_argument("--no-retrieval", action="store_true", help="send the whole document (full scan)")
    ap.add_argument("--highlight", metavar="FILE", help="also write a Markdown review copy: ~~old~~ **new**")
    ap.add_argument("--format", choices=["json", "text"], default="json",
                    help="json: status + stats + changes (default). text: only the updated document on stdout, "
                         "failures on stderr with exit code 1 (for shells and batch runners)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 (no document) if any instruction was skipped or changed nothing")
    ap.add_argument("-o", "--output", help="write the result here (default: stdout)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252

    def write_out(text: str):
        if args.output:
            tmp = args.output + ".tmp"
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            os.replace(tmp, args.output)
        else:
            sys.stdout.buffer.write(text.encode("utf-8"))   # no newline translation
            sys.stdout.flush()

    def fail(msg):
        if args.format == "text":
            print(f"error: {msg}", file=sys.stderr)
        else:
            write_out(json.dumps({"status": "error", "error": msg}, indent=2, ensure_ascii=False))
        return 1

    try:
        with open(args.input, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise UpdateFailed("input must be a JSON object.")
        doc, s = update_document(data.get("sourceDocument"), data.get("changes"), OpenAIClient(),
                                 retrieval=not args.no_retrieval)
        if args.strict and s.skipped:
            raise UpdateFailed("strict mode: " + "; ".join(f"#{k['instruction']} {k['reason']}" for k in s.skipped))
    except (OSError, json.JSONDecodeError) as e:
        return fail(f"cannot read input: {type(e).__name__}")
    except UpdateFailed as e:
        return fail(str(e))
    except Exception as e:  # unattended runs must always get a clean failure, never a traceback
        return fail(f"unexpected {type(e).__name__}")
    if args.highlight:
        with open(args.highlight, "w", encoding="utf-8") as f:
            f.write(s.highlighted)
    if args.format == "text":
        for k in s.skipped:
            print(f"skipped #{k['instruction']}: {k['reason']}", file=sys.stderr)
        write_out(doc)
        return 0
    changed = [c for k in sorted(s.changes) for c in s.changes[k]]
    write_out(json.dumps({
        "status": "ok", "updatedDocument": doc,
        "skipped": s.skipped,
        "stats": {"mode": s.mode, "wordsTotal": s.words_total, "wordsSent": s.words_sent,
                  "windows": s.chunks, "modelCalls": s.model_calls, "retries": s.retries,
                  "editsApplied": s.edits_applied, "promptTokens": s.prompt_tokens,
                  "completionTokens": s.completion_tokens},
        "changes": changed}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
