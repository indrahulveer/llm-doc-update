"""Cheap, LLM-free relevance selection: decide which parts of a document the model must see.

Query-focused filtering with BM25 over paragraph-aligned units (stdlib only).
  * Targeted instruction -> only the matching units (+ neighbours / the rest of a matched
    markdown section) are sent. Tokens and latency scale with the size of the *relevant
    region*, not the document.
  * Document-wide instruction (tone, style, "throughout", ...) or an instruction with no
    lexical hit at all -> every unit is sent (the safe, recall-first fallback).
"""
from __future__ import annotations

import math
import re
from collections import Counter

UNIT_WORDS = 250        # retrieval granularity (~330 tokens)
WINDOW_WORDS = 1500     # max words per model call after merging adjacent selected units
REL_THRESHOLD = 0.25    # keep units scoring >= 25% of the best unit (recall-biased)
SECTION_CAP_WORDS = 1200

GLOBAL_RE = re.compile(
    r"\b(throughout|entire|whole|everywhere|every (?:section|paragraph|sentence|page|mention|occurrence)"
    r"|all (?:sections|paragraphs|sentences|text|pages|mentions|occurrences|instances)|tone|formal|informal"
    r"|style|spelling|grammar|punctuation|capitali[sz]\w*|terminolog\w*|consisten\w*|voice|reading level"
    r"|translate|rewrite the (?:document|text))\b", re.I)

STOP = set("""a an the and or of to in on for with by at from as is are was were be been it its this that these
those into than then so such across about over under up down out if not no do does did can will should would
update remove delete change changes make add insert replace rewrite edit reflect increase decrease modify
section paragraph sentence text document line lines part parts mention mentions all any each every
please ensure also more less""".split())


def stem(w: str) -> str:
    for suf in ("ing", "edly", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if t not in STOP]


def bm25_scores(query: list[str], units: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    docs = [Counter(tokenize(u)) for u in units]
    lens = [sum(d.values()) for d in docs]
    avg = (sum(lens) / len(lens)) or 1.0
    n = len(units)
    df = Counter(t for d in docs for t in d)
    scores = []
    for d, ln in zip(docs, lens):
        s = 0.0
        for t in set(query):
            f = d.get(t, 0)
            if f:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * ln / avg))
        scores.append(s)
    return scores


def is_heading(unit: str) -> bool:
    return unit.lstrip().startswith("#")


def words(s: str) -> int:
    return len(s.split())


def select(units: list[str], changes: list[str]) -> tuple[list[set[int]], str]:
    """Return, per unit, the set of instruction indices it must be sent with; plus a mode label."""
    wanted = [set() for _ in units]
    modes = set()
    for i, ch in enumerate(changes):
        if GLOBAL_RE.search(ch):
            modes.add("global")
            for w in wanted:
                w.add(i)
            continue
        scores = bm25_scores(tokenize(ch), units)
        top = max(scores)
        if top <= 0:                       # vocabulary gave no signal: do not guess, scan everything
            modes.add("fallback-full")
            for w in wanted:
                w.add(i)
            continue
        modes.add("targeted")
        hits = {u for u, s in enumerate(scores) if s >= REL_THRESHOLD * top}
        keep = set(hits)
        for u in hits:
            keep.update(x for x in (u - 1, u + 1) if 0 <= x < len(units))   # neighbours as context
            if is_heading(units[u]):                                         # matched heading -> its section
                spent, v = 0, u + 1
                while v < len(units) and not is_heading(units[v]) and spent < SECTION_CAP_WORDS:
                    keep.add(v)
                    spent += words(units[v])
                    v += 1
        for u in keep:
            wanted[u].add(i)
    mode = "targeted" if modes == {"targeted"} else "+".join(sorted(modes))
    return wanted, mode


def plan(units: list[str], wanted: list[set[int]], window_words: int = WINDOW_WORDS):
    """Merge adjacent selected units into windows. Returns [(text, instr_set or None)] covering the doc."""
    segs: list[list] = []
    for text, w in zip(units, wanted):
        w = set(w) or None
        if segs and segs[-1][1] is None and w is None:
            segs[-1][0] += text
        elif (segs and w is not None and segs[-1][1] is not None
              and words(segs[-1][0]) + words(text) <= window_words):
            segs[-1][0] += text
            segs[-1][1] |= w
        else:
            segs.append([text, w])
    return [(t, w) for t, w in segs]
