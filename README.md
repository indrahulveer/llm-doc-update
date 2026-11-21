# LLM-Driven Document Updates

CLI that applies natural-language edit instructions to a document using the OpenAI API. Python 3.10+, standard library only (no installs).

## 1. Setup and commands

```powershell
$env:OPENAI_API_KEY = "<your key>"        # bash: export OPENAI_API_KEY=...
# optional: $env:OPENAI_MODEL = "gpt-4o-mini" (default)

python docupdate.py sample_input.json                 # result JSON on stdout
python docupdate.py sample_input.json -o result.json  # or to a file (atomic write)

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
{"status": "ok", "updatedDocument": "...", "stats": {"chunks": 7, "modelCalls": 7, "retries": 0,
  "editsApplied": 12, "promptTokens": 0, "completionTokens": 0}}
```

Failure (exit code 1). There is **no** `updatedDocument` field, and nothing partial is written:
```json
{"status": "error", "error": "chunk 3 failed after 3 attempts; last error: edits[0].find does not occur verbatim ..."}
```
Failure cases: a missing key, a bad input, persistent API errors, a chunk still invalid after retries, or a model that produced no effective change at all. A no-op is treated as failure because it's usually a silent miss. A batch runner can key on the exit code and `status`.

## 3. Architecture

**Alternative A: full-document rewrite.** Simple, and it is naturally good at global rewrites. But for a 10k-word document, output is about 13k tokens, which takes minutes. Long outputs get truncated or silently drift, and there is no cheap way to prove unrelated text was preserved (you'd have to diff and guess intent). A single failure costs the whole call.

**Chosen: B, edit proposals validated and applied by the program, per chunk.**
1. Split the document on blank-line boundaries into chunks of about 1,500 words (`''.join(chunks) == source` is asserted).
2. For each chunk, the model sees all instructions plus that chunk and returns `{"edits": [{"find", "replace"}]}`.
3. The program validates and applies the edits: `find` must occur verbatim exactly once, edits must not overlap, and the JSON must match the schema. Any violation sends the specific error back to the model for a corrected response (max 3 attempts per chunk, plus bounded backoff retries for 429/5xx/network errors).
4. If **any** chunk fails, the whole job fails. Chunks are only concatenated after all succeed.

I did not switch approaches during implementation. I picked B up front because it makes preservation of unrelated text true by construction and gives mechanical validation.

### Targeted vs document-wide requests
- **Targeted** ("remove the legacy API paragraph", "+10% pricing"): most chunks return `{"edits": []}`. Output tokens are tiny, and untouched text is byte-identical.
- **Document-wide** ("more formal"): the model must emit an edit for each affected paragraph. Chunking keeps each response bounded, and chunks run in parallel (4 workers), so no single call must produce 10k words. Accuracy is lower than a one-shot rewrite, because the model can't see other chunks. Each edited paragraph is rewritten independently, so style can vary slightly across chunks.
- **Cross-chunk instructions** ("make section 5 consistent with section 2", "reorder sections") are a limitation. A chunk can't see the others, so these may be missed or half-applied. Full rewrite would handle them better. This is the main accuracy cost of choosing B.

### Cost scaling (N words, chunk size C=1,500, k instructions)
- Calls: about N/C, so about 7 for 10k words.
- Input tokens: about 1.3N (the document, sent once in total) plus (N/C) × (system prompt + k instructions), which is linear in N.
- Output tokens: proportional to the *size of the edits*. It is near zero for targeted changes and up to about 1.3N for a global rewrite. Full rewrite (A) is always about 1.3N.
- Latency: roughly (N/C ÷ workers) × per-call latency, with each call bounded by chunk size. For A, latency grows with the full output. Each retry only re-costs one chunk.
- Edit quality is better with more context per call. Smaller chunks mean cheaper retries but more repeated instruction overhead and less context.

## 4. Verification actually run

| Check | Status |
|---|---|
| Unit tests (`python -m unittest discover -s tests -v`): 9 pass | **ran** |
| Targeted edit, unrelated text preserved | **ran** (stubbed model) |
| Malformed JSON, then not-found `find`, then success | **ran** (stubbed model) |
| Failure after retries exhausted (exactly 3 calls, bounded), one bad chunk fails the job, no-op fails | **ran** (stubbed model) |
| CLI exit code 1 and no document on failure (no key set) | **ran** |
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
- `docupdate.py`: chunking, validation, application, OpenAI client, CLI
- `sample_input.json`: example input
- `tests/test_docupdate.py`: offline tests with a scripted model stub
