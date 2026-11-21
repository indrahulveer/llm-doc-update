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
| 0b. Percent price change | "Raise/lower <scope> prices by N%" is exact arithmetic, done in code on prices written like `$40 per month` / `$400 per year`. Scope words narrow it: monthly vs annual (only the clause before `;` or `leave`) and tier names that appear in the document as `**Name**:`. If no such price is found it falls back to the LLM path | no |
| 1. Grounding pre-flight | Quoted targets must exist, and for replace-style instructions must be unambiguous (`'contact support'` occurring twice is skipped). Capitalised names on the target side of an edit verb must exist in the document ("Change the Platinum tier..." is skipped). For update/change/set, the content words of the target must exist ("support phone number" with no phone in the document is skipped) | no |
| 2. Targeted edits | BM25 query-focused filtering over paragraph units (about 250 words) selects the relevant windows, with neighbour and whole-section expansion and a full-scan fallback when nothing matches. The model returns `{change, find, replace, all}` edits, which are validated and applied | yes, relevant windows only |
| 3. Document-wide edits | Instructions about tone, style or "throughout" rewrite prose paragraphs, addressed **by paragraph id** (no verbatim anchors that can be mis-copied). Headings and code blocks are never offered | yes, all prose |

**Why stages 0 and 0b exist (a switch during implementation).** On the first full dataset run (486/500 pass, 7 wrong), the remaining wrong outputs were price edits: the model skipped one tier, raised the annual list when asked for monthly, or mangled a multi-paragraph anchor. Arithmetic with a clear scope does not need a language model, so it moved into code. That removed every price failure and the tokens they cost.

**Why stage 3 exists (a switch during implementation).** The first version used verbatim find/replace for tone rewrites as well. On the real dataset, tone cases failed repeatedly: with dozens of edits per window the model mis-copied anchors or produced non-unique ones, and three attempts were not enough. That triggered the switch to paragraph-id rewriting for document-wide instructions, with a programmatic check that numbers, prices, URLs, `inline code`, **bold** terms and list markers survive. Rename was moved out of the LLM entirely for the same reason (dozens of identical occurrences cannot be anchored uniquely).

### Validation (all programmatic)
- **Edits:** JSON schema, `change` is a valid instruction number, `find` occurs verbatim and exactly once (unless `all` is set), no overlapping edits, replacement size sanity, and a **relevance check on large deletions** (a deletion must share a content word with the instruction it claims to serve). That check came from a real failure where the model deleted an unrelated block.
- **Rewrites:** ids valid and unique, a single non-empty paragraph, length ratio sane, list/quote marker unchanged, no new headings or code fences, **protected text unchanged**. Each paragraph is judged on its own: a rejected paragraph is re-asked alone, and if it still fails it is **left unchanged and reported** under `skipped` instead of failing the whole document.
- **Marker boundaries:** an edit may not start in the middle of a markdown marker (an edit that began inside `### ` left a stray `#` behind in a real run).
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
| Accuracy | Apply intended changes, preserve the rest | 497/500 pass, 3 accepted clean failures, 0 wrong (section 5) |
| Reliability | Catch detectable errors, never accept silently | Met. Concrete catches found during live runs are listed in `WHAT_WAS_DONE.md` |
| Scalability | About 10k words, unattended, many documents | Met: `batch.py` ran all 500 documents (650-9,600 words) unattended in about 9 minutes |
| Latency | Reasonable as documents grow | Measured in section 5 |
| Token efficiency | Scales with relevant content | Measured in section 5 |

## 5. Evaluation on the 500-case dataset

Dataset: `doc-edit-dataset-500` (zip included). Command: `python batch.py <dataset> --out results --jobs 4`. Model: `gpt-4o-mini`. This is the final code, run on all 500 documents from scratch.

| Metric | Result |
|---|---|
| Documents succeeded / failed | 497 / 3 |
| Dataset scorer (`score.py`) | **497 pass, 3 clean_fail (accepted), 0 FAIL** |
| Prompt / completion tokens | 983,256 / 432,297 (about 2.0k / 0.9k per document on average) |
| Model calls | 1,458 in total (about 2.9 per document); 101 documents needed none |
| Text sent to the model | 499,197 of 1,520,353 words (33%) |
| Latency per document | mean 4.3s, p50 1.5s, p95 19.8s, max 40.6s (4 documents in parallel; wall clock 552s for all 500) |
| Documents over 8,000 words (67) | mean 9.2s, max 40.6s |
| Retries (bad model response corrected) | 26 |
| Documents where an instruction was skipped and reported | 352 (the dataset deliberately includes missing and ambiguous targets) |

The 3 clean failures are documents whose only requested change is the ambiguous `'contact support'` replacement. Nothing is applicable, so the tool exits 1 with no document, and the dataset's scorer accepts that.

Per change type (cases containing it): ambiguous_target 215 pass + 3 clean_fail, change_rate_limit 240, formal_tone 122, missing_target 214 + 3, raise_all_prices 103, raise_annual_only 90, raise_monthly_only 79, raise_pro_annual 98, remove_legacy_api 232, rename_product 229. Raw data: `eval/records_full_500.jsonl`, `eval/summary_full_500.json`, `eval/score_full_500.txt`.

**Token and latency scaling in this run.** Targeted edits send a roughly constant amount of text regardless of document size. Document-wide tone rewrites are linear in document size (these are the slow, 20-40s documents). Renames and price changes cost zero tokens.

**Caveats, stated plainly**
- **This is not a blind test.** I fixed bugs while looking at failures from this same dataset (iteration history: 6/12 and 9/12 early on, then 486/500 on the first full run, then 497/500 after the fixes). The fixes are general rules, but the score is partly tuned to this dataset's document shapes.
- The scorer checks tone cases only for surviving protected strings and headings. I read one rewritten document by eye (case_001): the prose was more formal, the rename applied, the code block untouched. I did not read the others.
- Only `gpt-4o-mini` was used, and the deterministic price and rename paths match the dataset's `$N per month/year` and "Rename X to Y" formats.

## 6. A semantically wrong response that passes every check

- **Source:** `Basic costs $10 per month. Pro costs $20 per month.`
- **Requested change:** `Increase the cost of every plan by 10%.` (worded without the word "price" so it reaches the model; a request like "Raise all prices by 10%" is computed exactly in code and cannot go wrong this way)
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
0. **The price and rename shortcuts are format-bound.** Percent price changes are exact only for prices written `$N per month` or `$N per year`; other price formats go to the LLM, which can still get arithmetic or scope wrong (section 6). Renames are literal, so a rename that needs judgement is not handled deterministically.
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
- `tests/test_docupdate.py`: 31 offline tests with a scripted model stub
- `sample_input.json`: example input
- `eval/`: records, summary and scorer output from the live full-dataset run
- `WHAT_WAS_DONE.md`: summary of the work and of the bugs found and fixed
