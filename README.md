# LLM-Driven Document Updates

CLI that applies natural-language edit instructions to a document using the OpenAI API. Python 3.10+, standard library only (no installs). See `WHAT_WAS_DONE.md` for a plain summary of the work and the bugs found and fixed along the way.

## 1. Setup and commands

```powershell
$env:OPENAI_API_KEY = "<your key>"        # bash: export OPENAI_API_KEY=...
# optional: $env:OPENAI_MODEL = "gpt-4o-mini" (default)

# one document
python docupdate.py sample_input.json                      # JSON result on stdout
python docupdate.py sample_input.json --format text        # only the updated document on stdout
python docupdate.py sample_input.json --highlight review.md  # also write a ~~old~~ **new** review copy
python docupdate.py sample_input.json --strict             # exit 1 if any instruction was skipped
python docupdate.py sample_input.json --no-retrieval       # send the whole document (baseline)

# many documents, unattended (continues after failures, prints a final report, runs score.py if present)
python batch.py <dataset_or_inputs_dir> --out results --jobs 4 [--limit N] [--resume]

# tests (no API key needed)
python -m unittest discover -s tests -v
```

The key is read only from `OPENAI_API_KEY`. It is never written to output or logs, and API error bodies are not logged.

## 2. Input, output, failure

Input: `{"sourceDocument": "full text...", "changes": ["Make the tone more formal."]}`

Success (exit code 0), `--format json` (default):
```json
{"status": "ok", "updatedDocument": "...",
 "skipped": [{"instruction": 2, "text": "...", "reason": "target 'Data Residency' not found in the document"}],
 "stats": {"mode": "targeted+rewrite", "wordsTotal": 5039, "wordsSent": 2100, "windows": 9, "modelCalls": 9,
           "retries": 0, "editsApplied": 31, "promptTokens": 0, "completionTokens": 0},
 "changes": [{"before": "Basic costs $10", "after": "Basic costs $11"}]}
```
`--format text` prints only the document (skips are listed on stderr), which suits shell pipelines and the dataset's `run_batch.py`.

Failure (exit code 1). There is **no** document in the output, and nothing partial is written:
```json
{"status": "error", "error": "rewrite window 1000002 failed after 3 attempts; last error: [4] lost protected text ..."}
```
Failure cases: missing key, bad input, persistent API errors, a window still invalid after retries, no effective change at all, no instruction resolvable, or (with `--strict`) any skipped instruction.

**Skipped instructions.** If an instruction's target is missing or ambiguous it is **not applied**, and it is listed under `skipped` with a reason. The rest of the instructions are applied. This is an explicit, reported decision, not a silent one. `--strict` turns any skip into a failure.

## 3. Architecture

**Alternative A: full-document rewrite.** Simple, and it is naturally good at global rewrites. But for a 10k-word document the output is about 13k tokens (minutes of latency), long outputs get truncated or silently drift, and there is no cheap way to prove unrelated text was preserved. One failure costs the whole call. Rejected.

**Chosen: staged pipeline. Cheap deterministic NLP first, the LLM only where it is needed, and programmatic validation of everything the LLM returns.**

| Stage | What it does | LLM? |
|---|---|---|
| 0. Literal rename | "Rename X to Y everywhere" is a literal substitution, done in code. It also renames derived forms (joined/hyphenated lowercase, as in URLs and code) | no |
| 1. Grounding pre-flight | Quoted targets must exist, and for replace-style instructions must be unambiguous (`'contact support'` occurring twice is skipped). Capitalised names on the target side of an edit verb must exist in the document ("Change the Platinum tier..." is skipped). For update/change/set, the content words of the target must exist ("support phone number" with no phone in the document is skipped) | no |
| 2. Targeted edits | BM25 query-focused filtering over paragraph units (about 250 words) selects the relevant windows, with neighbour and whole-section expansion and a full-scan fallback when nothing matches. The model returns `{change, find, replace, all}` edits, which are validated and applied | yes, relevant windows only |
| 3. Document-wide edits | Instructions about tone, style or "throughout" rewrite prose paragraphs, addressed **by paragraph id** (no verbatim anchors that can be mis-copied). Headings and code blocks are never offered | yes, all prose |

**Why stage 3 exists (a switch during implementation).** The first version used verbatim find/replace for tone rewrites as well. On the real dataset, tone cases failed repeatedly: with dozens of edits per window the model mis-copied anchors or produced non-unique ones, and three attempts were not enough. That triggered the switch to paragraph-id rewriting for document-wide instructions, with a programmatic check that numbers, prices, URLs, `inline code`, **bold** terms and list markers survive. Rename was moved out of the LLM entirely for the same reason (dozens of identical occurrences cannot be anchored uniquely).

### Validation (all programmatic)
- **Edits:** JSON schema, `change` is a valid instruction number, `find` occurs verbatim and exactly once (unless `all` is set), no overlapping edits, replacement size sanity, and a **relevance check on large deletions** (a deletion must share a content word with the instruction it claims to serve). That check came from a real failure where the model deleted an unrelated block.
- **Rewrites:** ids valid and unique, a single non-empty paragraph, length ratio sane, list/quote marker unchanged, **protected text unchanged**.
- **Job level:** every instruction is attributed to edits. An instruction that changed nothing is reported in `skipped`. A job that changes nothing at all fails.
- **Deletions** collapse the blank-line run they would leave behind.

### Recovery
Up to 3 attempts per window, feeding the specific error back to the model, plus backoff for 429/5xx/network errors (honouring `Retry-After`). Any window that is still bad fails the whole document. No partial document is returned. `batch.py` records the failure and moves on to the next document.

### Cost scaling (N = document words)
- **Targeted:** about 2-3 units (around 560 words) per matched region. This is constant in N for a fixed number of matches, usually one call.
- **Document-wide:** linear in N. Input about 1.3N tokens, output up to about 1.3N, in windows of about 350 words, 4 in parallel.
- **Rename:** zero tokens.
- **Full rewrite (A):** about 1.3N in and 1.3N out for every request, in one serial call.
- Retries only re-cost the one failed window.

### Which NLP methods are used, and which are not
| Method | Used? | Notes |
|---|---|---|
| Normalisation, tokenisation, stop-word removal, light stemming | yes, **for retrieval and grounding only** | The text sent to the LLM is never altered |
| Paragraph segmentation, markdown-heading structure | yes | Units, sections and rewrite windows |
| BM25 query-focused filtering | yes | The core token saver for targeted edits |
| Rule-based instruction parsing (regex): rename, quoted anchors, capitalised entities, global-intent markers | yes | Routing and grounding |
| Embeddings, rerankers, NER models, coreference, topic models, zero-shot classification | no | Extra dependencies for no clear gain on this task |
| Summarisation and prompt compression | **no, on purpose** | They would corrupt the text that must be copied or returned |

## 4. Requirements compliance

Status: **Met** = implemented and covered by offline tests; **Live** = also exercised against the real OpenAI API on the dataset (section 5).

| ID | Requirement | Status |
|---|---|---|
| FR-1 | Accept `sourceDocument` and `changes` | Met, Live |
| FR-2 | LLM interprets instructions and produces edits | Met, Live |
| FR-3 | Targeted and document-wide changes | Met, Live (price, rate-limit, removal, rename, tone cases) |
| FR-4 | Return the complete updated document | Met, Live |
| FR-5 | Validate model output | Met, Live (see Validation) |
| FR-6 | Recover, or fail explicitly so the batch continues | Met, Live. `batch.py` continues after failures. Exit code 1 with no document on failure |
| Accuracy | Apply intended changes, preserve the rest | 55/55 on the dataset subset run (section 5) |
| Reliability | Catch detectable errors, never accept silently | Met. Concrete catches found during live runs are listed in `WHAT_WAS_DONE.md` |
| Scalability | About 10k words, unattended, many documents | Met by `batch.py`. Documents up to 9.6k words are in the dataset, but only the first 55 documents were run |
| Latency | Reasonable as documents grow | Measured in section 5 |
| Token efficiency | Scales with relevant content | Measured in section 5 |

## 5. Evaluation on the 500-case dataset

The dataset is `doc-edit-dataset-500` (zip included in the submission). Run with `python batch.py <dataset> --out results --jobs 4 --resume`, model `gpt-4o-mini`.

**The full 500-case run was started and then stopped by request after 55 documents had completed. The numbers below cover those 55 documents only. They are not a 500-case result.**

| Metric | Result (first 55 documents) |
|---|---|
| Documents succeeded / failed | 55 / 0 |
| Dataset scorer (`score.py`) | **55 pass, 0 clean_fail, 0 FAIL** |
| Prompt / completion tokens | 124,393 / 48,033 (about 2.3k / 0.9k per document) |
| Text sent to the model | 62,145 of 160,535 words (39%) |
| Latency per document | mean 6.1s, p50 3.2s, p95 23.4s, max 38.0s (4 documents in parallel) |
| Documents where some instruction was skipped and reported | 43 (the dataset deliberately includes missing and ambiguous targets) |

Per change type, all passing: ambiguous_target 28, change_rate_limit 21, formal_tone 14, missing_target 25, raise_all_prices 10, raise_annual_only 11, raise_monthly_only 9, raise_pro_annual 12, remove_legacy_api 30, rename_product 28 (cases containing each type). The raw per-document records are in `eval/records.jsonl` and the scorer output in `eval/partial_score.txt`.

Caveats:
- The scorer checks tone cases only for surviving protected strings and headings. I read one rewritten document by eye (case_001): the prose became more formal, the rename was applied, and the code block was untouched. I did not read the others.
- Earlier iterations of the code scored worse (for example 6/12 and 9/12 before the fixes in section 3). The final numbers come from the final code, but the fixes were developed against these same first cases, so they are not a blind test.
- Documents 56-500 were not run.

## 6. A semantically wrong response that passes every check

- **Source:** `Basic costs $10 per month. Pro costs $20 per month.`
- **Requested change:** `Increase all prices by 10%.`
- **Model response (valid JSON, both `find`s unique, no overlap):**
  ```json
  {"edits": [{"find": "Basic costs $10", "replace": "Basic costs $11"},
             {"find": "Pro costs $20",   "replace": "Pro costs $24"}]}
  ```
- **Expected result:** `Basic costs $11 per month. Pro costs $22 per month.`
- **Actual result:** `Basic costs $11 per month. Pro costs $24 per month.`

`test_readme_semantic_failure_example` feeds this response through the real `update_document` path and asserts that the wrong `$24` is accepted.

**What the checks cannot establish:** they are structural. They show the response is well-formed, anchors exist, edits are disjoint, protected text survived, and a deletion shares words with its instruction. They cannot show an edit is *correct*: that arithmetic is right, that the right passages were targeted, that all relevant passages were found, or that tone or meaning is good. The `--highlight` review copy and the `changes` list exist so a human can verify quickly. A second model pass or recomputing numbers would close more of the gap, and neither is implemented.

## 7. Known limitations
1. **Silent retrieval misses.** A passage that needs a targeted edit but shares no vocabulary with the instruction is not sent to the model and is skipped without an error. `--no-retrieval` is the safe fallback.
2. **Grounding heuristics can false-skip.** An instruction like "Change the Pro price to $25" is skipped if the document never uses the word "price" (it says "costs"). Skips are always reported, and `--strict` makes them failures, but a legitimate instruction can be skipped.
3. **Skip-and-continue by default.** A document with a skipped instruction exits 0 with the rest applied. Use `--strict` if a partial result is unacceptable.
4. **Tone quality is not verified**, only that facts, numbers, links, code, headings and list markers survive.
5. **Instruction interplay.** Instructions are processed in stages (rename, targeted, document-wide), not strictly in the order given.
6. **Rename highlighting.** Literal renames are reported in `changes` with an occurrence count, not marked up inline in the `--highlight` copy.
7. **Cross-window references** ("make section 5 consistent with section 2") are weak, because a window cannot see the others.

## Files
- `docupdate.py`: pipeline, validation, OpenAI client, CLI
- `selector.py`: BM25 relevance selection and window planning (no LLM)
- `batch.py`: unattended multi-document runner with a final report and optional dataset scoring
- `tests/test_docupdate.py`: 28 offline tests with a scripted model stub
- `sample_input.json`: example input
- `eval/`: records and scorer output from the live dataset run
- `WHAT_WAS_DONE.md`: summary of the work and of the bugs found and fixed
