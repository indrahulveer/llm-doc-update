# What was done

## Goal
A CLI that takes `{"sourceDocument", "changes"}`, uses the OpenAI API to apply natural-language edits, validates what the model returns, recovers from bad responses, and fails explicitly (exit code 1, no document) when it cannot. It was then tested live against the 500-case dataset.

## Built
1. **First version.** The model proposes `find`/`replace` edits, and the program validates and applies them. There is bounded retry, with the error fed back to the model.
2. **BM25 retrieval stage** (`selector.py`) so that targeted edits send only the relevant part of the document. Cost for targeted edits does not grow with document size.
3. **Change highlighting.** The JSON output has a `changes` list, and `--highlight` writes a `~~old~~ **new**` review copy.
4. **Hardening.** UTF-8 output on Windows, a clean JSON failure for any unexpected error, and no raw tracebacks.
5. **Dataset integration.**
   - `--format text`: only the document goes to stdout, so the dataset's `run_batch.py` and shell pipelines work.
   - `--strict`: any skipped instruction becomes a failure.
   - `batch.py`: unattended runner that continues after failures, resumes after interruption (`--resume`), prints a final report (success/failure counts, tokens, latency percentiles, text sent) and runs the dataset's `score.py`.
6. **Per-instruction attribution.** Every edit names the instruction it serves, so an instruction that changed nothing is reported instead of silently ignored.
7. **Grounding pre-flight** (no LLM): missing or ambiguous quoted targets, capitalised names that are not in the document, and update/change/set targets that are not in the document are skipped and reported.
8. **Deterministic literal rename** (no tokens), including derived forms such as `nimbusdesk` in URLs.
9. **Deterministic percent price change** (monthly / annual / named tier scope) for `$N per month|year` prices, with exact decimal arithmetic.
10. **Per-paragraph recovery in the rewrite stage:** a paragraph that still fails validation is left unchanged and reported instead of failing the document.
11. **Paragraph-id rewrite stage** for tone and other document-wide instructions, with programmatic checks that numbers, prices, URLs, code, bold terms and list markers survive.

## Bugs found by running the real dataset, and the fixes
| Problem seen in live runs | Fix |
|---|---|
| Tone and rename cases failed after 3 attempts: the model mis-copied or duplicated verbatim anchors across dozens of edits | Rename became deterministic code. Tone moved to paragraph-id rewriting with protected-token checks |
| For "Change the Platinum tier price to $999" (no such tier) the model edited a different tier | Capitalised-name grounding: skip and report |
| For "Update the support phone number" (no phone number exists) the model inserted one | Target-word grounding for update/change/set: skip and report |
| "Replace the phrase 'contact support'" (appears twice) was applied to both | Quoted-anchor ambiguity check: skip and report, before any API call |
| Over-selection: a stray "api" word pulled an unrelated window in, and the model invented a 42-line deletion | BM25 threshold raised from 25% to 40% of the best unit, a better stemmer (price/pricing/prices), and a relevance check that rejects large deletions sharing no word with the instruction |
| Deleting a section left an extra blank line | Deletions collapse the blank-line run they leave |
| Model calls failed on repeated identical sentences in the generated documents (non-unique anchors) | Duplicates are allowed only through the explicit `all` flag or the rewrite stage |
| First full run: price edits wrong (a tier skipped, annual raised instead of monthly, mangled multi-paragraph anchor) | Percent price changes moved into code (exact arithmetic, scoped) |
| A rewrite window failed the whole document because one paragraph dropped a number or collapsed repeated text | Per-paragraph rejection: re-ask that paragraph, else leave it unchanged and report it |
| A deletion edit began inside a `###` heading marker and left a stray `#` | Edits may not start in the middle of a markdown marker |
| Rate-limit retries too short | 5 retries, honouring `Retry-After` |
| Windows console could not print non-ASCII output | UTF-8 stdout, raw bytes for `--format text` |

## Verified
- **31 offline unit tests pass.** They cover targeted edits, malformed responses recovering, failure after retries are exhausted, one bad window failing the whole job, no-op failure, retrieval scaling, grounding checks, rename, price arithmetic and scope, rewrite protection and per-paragraph recovery, heading-marker boundaries, highlighting, the CLI failure exit code, and the batch runner continuing after a failure.
- **Live run on all 500 dataset documents (final code, from scratch):** 497 pass, 3 clean failures (accepted by the scorer), 0 wrong outputs. 983k prompt and 432k completion tokens in total, 33% of the text sent to the model, mean 4.3s and p95 19.8s per document, 552s wall clock with 4 documents in parallel.
- **History of the live runs:** the first full run scored 486 pass, 7 clean_fail, 7 wrong. The causes were fixed (see the table above) and the full set was re-run.

## Not verified, stated plainly
- **Not a blind test.** The fixes were made while looking at failures from this same dataset.
- The dataset scorer checks tone cases only for protected strings and headings. I read one tone output by eye (it looked right) and did not read the rest.
- The grounding heuristics can skip a legitimate instruction (always reported, never silent).
- The deterministic price path only understands `$N per month|year`, and the rename path only literal renames.
- Only `gpt-4o-mini` was used.

## Security
The API key is read only from `OPENAI_API_KEY`. During testing it was loaded from your key file inside the shell command and never printed or written. It is not in the repo, the zip, the logs, or the results.
