import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import docupdate as d

ROOT = os.path.join(os.path.dirname(__file__), "..")


class ScriptedClient:
    """Returns canned responses in order; records how many calls were made."""
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), 0

    def complete(self, messages):
        self.calls += 1
        return self.responses.pop(0), {"prompt_tokens": 10, "completion_tokens": 5}


def edits(*pairs):
    return json.dumps({"edits": [{"find": f, "replace": r} for f, r in pairs]})


with open(os.path.join(ROOT, "sample_input.json"), encoding="utf-8") as _f:
    SAMPLE = json.load(_f)
GOOD = edits(("Basic costs $10", "Basic costs $11"), ("Pro costs $20", "Pro costs $22"),
             ("Enterprise costs $100", "Enterprise costs $110"),
             ("## Legacy API\n\nThe legacy API (v1) is deprecated and will be removed next year.\n\n", ""))


class Tests(unittest.TestCase):
    def run_doc(self, client, **kw):
        return d.update_document(SAMPLE["sourceDocument"], SAMPLE["changes"], client, workers=1, **kw)

    def test_targeted_edit_preserves_unrelated_text(self):
        doc, stats = self.run_doc(ScriptedClient(GOOD))
        src = SAMPLE["sourceDocument"]
        self.assertIn("Basic costs $11", doc)
        self.assertNotIn("legacy", doc.lower())
        self.assertTrue(doc.endswith("## Support\n\nContact us anytime, we're happy to help you out!\n"))
        self.assertTrue(doc.startswith(src[:src.index("Basic")]))
        self.assertEqual((stats.model_calls, stats.retries), (1, 0))

    def test_malformed_then_recovery(self):
        client = ScriptedClient("not json {", edits(("text that is not there", "x")), GOOD)
        doc, stats = self.run_doc(client)
        self.assertIn("Pro costs $22", doc)
        self.assertEqual((client.calls, stats.retries), (3, 2))

    def test_ambiguous_and_overlapping_edits_rejected(self):
        with self.assertRaises(d.InvalidEdits):
            d.apply_edits("a b a", [("a", "x")])
        with self.assertRaises(d.InvalidEdits):
            d.apply_edits("hello world", [("hello wor", "x"), ("world", "y")])

    def test_failure_after_recovery_exhausted(self):
        client = ScriptedClient("garbage", "{}", '{"edits": [{"find": "nope", "replace": "x"}]}')
        with self.assertRaises(d.UpdateFailed) as cm:
            self.run_doc(client)
        self.assertEqual(client.calls, d.MAX_ATTEMPTS)  # bounded
        self.assertIn("failed after", str(cm.exception))

    def test_one_bad_chunk_fails_whole_job(self):
        src = "para one\n\npara two\n\npara three"
        client = ScriptedClient(edits(("one", "1")), "bad", "bad", "bad")
        with self.assertRaises(d.UpdateFailed):
            d.update_document(src, ["x"], client, chunk_words=2, workers=1)

    def test_no_op_is_not_success(self):
        with self.assertRaises(d.UpdateFailed):
            self.run_doc(ScriptedClient(edits()))

    def test_chunking_roundtrip(self):
        src = "\n\n".join(f"paragraph {i} " + "word " * 50 for i in range(40))
        chunks = d.chunk_document(src, 300)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), src)

    def test_cli_failure_exit_code_and_no_document(self):
        out = os.path.join(ROOT, "tests", "_out.json")
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            code = d.main([os.path.join(ROOT, "sample_input.json"), "-o", out])
            with open(out, encoding="utf-8") as f:
                res = json.load(f)
        finally:
            os.path.exists(out) and os.remove(out)
            if old:
                os.environ["OPENAI_API_KEY"] = old
        self.assertEqual(code, 1)
        self.assertEqual(res["status"], "error")
        self.assertNotIn("updatedDocument", res)

    def test_readme_semantic_failure_example(self):
        """A wrong-but-valid response passes every check (documented in README)."""
        src = "Basic costs $10 per month. Pro costs $20 per month."
        resp = edits(("Basic costs $10", "Basic costs $11"), ("Pro costs $20", "Pro costs $24"))
        doc, _ = d.update_document(src, ["Increase all prices by 10%."], ScriptedClient(resp), workers=1)
        self.assertEqual(doc, "Basic costs $11 per month. Pro costs $24 per month.")  # expected $22


if __name__ == "__main__":
    unittest.main()
