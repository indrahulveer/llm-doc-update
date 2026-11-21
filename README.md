# LLM-Driven Document Updates

CLI that applies natural-language edit instructions to a document using the OpenAI API. Python 3.10+, standard library only (no installs).

## 1. Setup and commands

```powershell
$env:OPENAI_API_KEY = "<your key>"        # bash: export OPENAI_API_KEY=...
# optional: $env:OPENAI_MODEL = "gpt-4o-mini" (default)

python docupdate.py sample_input.json                 # result JSON on stdout
python docupdate.py sample_input.json -o result.json  # or to a file (atomic write)
python docupdate.py sample_input.json --no-retrieval  # force a full-document scan
python docupdate.py sample_input.json --highlight review.md   # also write a Markdown review copy of the changes

python -m unittest discover -s tests -v               # tests; no API key needed
```

The key is read only from `OPENAI_API_KEY`. It is never written to output or logs, and no HTTP error bodies are logged.

## 2. Input, output, failure

Input:
```json
{"sourceDocument": "full text...", "changes": ["Make the tone more formal."]}
```

Success (exit code 0):
```json
{"status": "ok", "updatedDocument": "...", "stats": {"mode": "targeted", "wordsTotal": 9931, "wordsSent": 558, "windows": 1, "modelCalls": 1, "retries": 0,
  "editsApplied": 12, "promptTokens": 0, "completionTokens": 0},
 "changes": [{"before": "Basic costs $10", "after": "Basic costs $11"}]}
```

Failure (exit code 1). There is **no** `updatedDocument` field, and nothing partial is written:
```json
{"status": "error", "error": "chunk 3 failed after 3 attempts; last error: edits[0].find does not occur verbatim ..."}
```
Failure cases: a missing key, a bad input, persistent API errors, a chunk still invalid after retries, or a model that produced no effective change at all. A no-op is treated as failure because it's usually a silent miss. A batch runner can key on the exit code and `status`.

## Techniques used (summary)

| Stage | Technique | Kind | LLM tokens? |
|---|---|---|---|
| Segmentation | Paragraph-aligned units (about 250 words) | rule-based NLP | none |
| Normalisation | Lowercasing, tokenisation, stop-word and editing-verb removal, light suffix stemming | classical NLP | none |
| Relevance | **BM25** ranking of units against each instruction (query-focused filtering) | information retrieval | none |
| Context | Neighbour units, and whole-section expansion from matched markdown headings | document-structure heuristics | small |
| Routing | Regex markers for document-wide instructions (tone, style, "throughout", ...), and zero-hit fallback to a full scan | rule-based | none |
| Editing | LLM proposes `{find, replace}` edits (structured JSON output, temperature 0) | LLM | only relevant windows |
| Verification | Schema check, verbatim-and-unique anchor match, overlap check, size sanity check | programmatic validation | none |
| Recovery | Error-feedback retry (max 3 per window) and backoff for 429/5xx/network | bounded retry | only the failed window |
| Review | Change highlighting (see below) | diff-style markup | none |

## Highlighting changes

Every run reports exactly what changed. This comes from the validated edits themselves, not from diffing afterwards.
- The JSON output has a `changes` list of `{before, after}` pairs (no-op edits are dropped).
- `--highlight review.md` writes the full document with each change marked as `~~removed~~` and `**added**`. Each line is marked separately, so multi-paragraph deletions render correctly in Markdown.

Output for `sample_input.json`:

```markdown
## Pricing

~~Basic costs $10~~**Basic costs $11** per month. ~~Pro costs $20~~**Pro costs $22** per month. ~~Enterprise costs $100~~**Enterprise costs $110** per month.

~~## Legacy API~~

~~The legacy API (v1) is deprecated and will be removed next year.~~
```

This lets a human verify the change in seconds, which is also the best defence against the semantically wrong responses described in section 5.

## 3. Architecture

**Alternative A: full-document rewrite.** Simple, and it is naturally good at global rewrites. But for a 10k-word document, output is about 13k tokens, which takes minutes. Long outputs get truncated or silently drift, and there is no cheap way to prove unrelated text was preserved (you'd have to diff and guess intent). A single failure costs the whole call.

**Chosen: B (validated edit proposals) plus classical-NLP retrieval in front of the LLM.**

Pipeline:
1. **Split** into paragraph-aligned units of about 250 words (`''.join(units) == source` is asserted).
2. **Select (no LLM, stdlib only, `selector.py`).** This is query-focused filtering:
   - Each instruction is tokenised (stop-words and generic editing verbs like "update" or "remove" dropped, light stemming) and scored against every unit with **BM25**.
   - Units scoring at least 25% of the best unit are kept. The threshold is recall-biased.
   - Neighbouring units are added as context. If a markdown heading matches, the rest of its section is added (up to about 1,200 words).
   - **Document-wide** instructions (tone, style, "throughout", spelling, and similar markers) select every unit.
   - An instruction with **no lexical hit** also selects every unit. We never guess "nothing relevant"; we fall back to a full scan.
3. **Merge** adjacent selected units into windows of up to 1,500 words. Each window is sent with **only the instructions that selected it**.
4. **Propose:** the model returns `{"edits": [{"find", "replace"}]}` for its window.
5. **Validate and apply:** `find` must occur verbatim exactly once, edits must not overlap, and the schema must match. A violation sends the specific error back to the model (max 3 attempts per window, plus bounded backoff for 429/5xx/network errors).
6. Unselected text is passed through byte-for-byte. If **any** window fails, the job fails. Nothing is assembled until all windows succeed.

I did not switch approaches during implementation. I added the retrieval stage after re-reading the requirement to scale token usage with *relevant* content rather than document size.

`--no-retrieval` forces the full scan, which is useful as an accuracy baseline.

### Which NLP methods from the list I used
| Method | Used? | Why |
|---|---|---|
| Normalisation, tokenisation, stop-word removal, light stemming | yes, **for retrieval only** | The text sent to the LLM is the original, unmodified |
| Paragraph segmentation | yes | Keeps units self-contained so `find` anchors stay valid |
| BM25 query-focused filtering | yes | The core token saver for targeted edits |
| Section-aware expansion (heading structure) | yes | "Pricing section" must include the tiers beneath the heading |
| Keyword fallback to full scan | yes | Protects recall |
| Embeddings, cross-encoder rerank | no | Needs model downloads or an extra API. BM25 is enough for edit instructions, which reuse the document's own wording |
| Extractive/abstractive summarisation, prompt compression (LLMLingua) | **no, on purpose** | Compressing the text would corrupt the exact text the model must copy into `find` and the output we return |
| NER, coreference, topic models, zero-shot classification | no | No measurable benefit for this task, and heavy dependencies |

### Targeted vs document-wide requests
- **Targeted** ("remove the legacy API paragraph", "+10% pricing"): only about 2-3 units are sent, usually one call. The rest of the document is untouched and byte-identical.
- **Document-wide** ("more formal"): all windows are sent, in parallel (4 workers). Each response is bounded by the window size.
- **Mixed requests:** each window receives only the instructions that selected it.
- **Accuracy trade-offs:**
  - The retrieval stage can **miss** a passage that needs editing but doesn't share vocabulary with the instruction (for example, "rename our flagship product" when the product is never named in the instruction). That is a silent miss. The mitigations are the recall-biased threshold, neighbour and section expansion, and the zero-hit fallback. A window can't see other windows, so cross-reference instructions ("make section 5 consistent with section 2") are weaker than a full rewrite.
  - Words in the instruction that also appear in unrelated paragraphs (a number like "11") can pull in extra units. That costs tokens, not correctness.

### Measured scaling (offline, synthetic doc; `words_sent` is what goes to the model)
| Document words | Targeted: words sent / windows | Global: words sent / windows |
|---|---|---|
| ~1.7k | 282 / 1 | 1,741 / 2 |
| ~9.9k | 558 / 1 | 9,931 / 7 |
| ~40k | 558 / 1 | 39,961 / 28 |

Input tokens, output tokens and latency by request type (N = document words):
- **Targeted:** input is about 1.3 × (relevant region), which is **constant in N**. Output is the size of the edits. Latency is one or two calls. The only O(N) cost is the local BM25 pass (milliseconds, no tokens).
- **Global:** input is about 1.3N plus a small per-window instruction overhead. Output is up to about 1.3N. Calls ≈ N/1500, run 4-way parallel, so latency ≈ ceil(N/1500/4) × per-call latency, and each call is bounded.
- **Full rewrite (A):** input about 1.3N and output about 1.3N for every request type, with a single long serial call.
- Retries only re-cost the one failed window.

## 4. Verification actually run

| Check | Status |
|---|---|
| Unit tests (`python -m unittest discover -s tests -v`): 15 pass | **ran** |
| Targeted edit, unrelated text preserved | **ran** (stubbed model) |
| Malformed JSON, then not-found `find`, then success | **ran** (stubbed model) |
| Failure after retries exhausted (exactly 3 calls, bounded), one bad chunk fails the job, no-op fails | **ran** (stubbed model) |
| CLI exit code 1 and no document on failure (no key set) | **ran** |
| Retrieval: targeted edit sends under 15% of a 5k-word doc, `words_sent` is constant as the doc grows 4x, global instruction sends every window, zero-hit falls back to full scan, heading pulls in its section | **ran** (stubbed model) |
| Highlight/changes output: markup is correct, and stripping it reproduces the updated document | **ran** (stubbed model) |
| Live OpenAI smoke test | **not run**: no `OPENAI_API_KEY` was configured in this environment |

The live path (`OpenAIClient`) is therefore untested against the real API. Run `python docupdate.py sample_input.json` with a key to check it.

## 5. A semantically wrong response that passes every check

- **Source:** `Basic costs $10 per month. Pro costs $20 per month.`
- **Requested change:** `Increase all prices by 10%.`
- **Model response (valid JSON, both `find`s unique, no overlap):**
  ```json
  {"edits": [{"find": "Basic costs $10", "replace": "Basic costs $11"},
             {"find": "Pro costs $20",   "replace": "Pro costs $24"}]}
  ```
- **Expected result:** `Basic costs $11 per month. Pro costs $22 per month.`
- **Actual result:** `Basic costs $11 per month. Pro costs $24 per month.`

This is verified by `test_readme_semantic_failure_example`, which feeds the response through the real `update_document` path (`parse_edits` → `apply_edits`) and asserts the wrong `$24` output is accepted.

**What the checks cannot establish:** the checks are all *structural*. They show the response is well-formed, that each anchor exists once, and that edits are disjoint, so the change is mechanically applicable and localized. They cannot show the edit is *correct*: that arithmetic is right, that the right passages were targeted, that all relevant passages were found (misses look the same as "nothing to change"), that tone or meaning was preserved, or that cross-chunk consistency holds. Closing that gap would need a second model pass or human review of the diff, domain-specific validators (e.g. recompute numbers), or sampling and agreement checks. None of these are implemented.

## Files
- `selector.py`: BM25 relevance selection and window planning (no LLM)
- `docupdate.py`: chunking, validation, application, OpenAI client, CLI
- `sample_input.json`: example input
- `tests/test_docupdate.py`: offline tests with a scripted model stub
