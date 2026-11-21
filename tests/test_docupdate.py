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

    def test_highlighted_copy_and_changes_list(self):
        doc, stats = self.run_doc(ScriptedClient(GOOD))
        self.assertIn("~~Basic costs $10~~**Basic costs $11**", stats.highlighted)
        self.assertIn("~~Pro costs $20~~**Pro costs $22**", stats.highlighted)
        self.assertEqual(stats.changes[0][0], {"before": "Basic costs $10", "after": "Basic costs $11"})
        # stripping the deleted text and the markers must give the updated document (modulo whitespace)
        import re
        plain = re.sub(r"~~.*?~~", "", stats.highlighted).replace("**", "")
        self.assertEqual(" ".join(plain.split()), " ".join(doc.split()))
        self.assertIn("~~## Legacy API~~", stats.highlighted)

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
            d.update_document(src, ["x"], client, unit_words=2, window_words=2, workers=1)

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

    def test_unexpected_error_still_yields_clean_failure(self):
        out = os.path.join(ROOT, "tests", "_out2.json")
        orig = d.OpenAIClient
        d.OpenAIClient = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            code = d.main([os.path.join(ROOT, "sample_input.json"), "-o", out])
            with open(out, encoding="utf-8") as f:
                res = json.load(f)
        finally:
            d.OpenAIClient = orig
            os.path.exists(out) and os.remove(out)
        self.assertEqual((code, res["status"]), (1, "error"))
        self.assertNotIn("updatedDocument", res)

    def test_readme_semantic_failure_example(self):
        """A wrong-but-valid response passes every check (documented in README)."""
        src = "Basic costs $10 per month. Pro costs $20 per month."
        resp = edits(("Basic costs $10", "Basic costs $11"), ("Pro costs $20", "Pro costs $24"))
        doc, _ = d.update_document(src, ["Increase all prices by 10%."], ScriptedClient(resp), workers=1)
        self.assertEqual(doc, "Basic costs $11 per month. Pro costs $24 per month.")  # expected $22


class RetrievalTests(unittest.TestCase):
    NL = chr(10)

    def long_doc(self, n=60):
        nl = self.NL * 2
        paras = [f"## Topic {i}{nl}Filler paragraph number {i} about gardening and weather. " + "lorem ipsum " * 40
                 for i in range(n)]
        paras[min(37, n - 1)] = f"## Pricing{nl}Basic costs $10 per month. Pro costs $20 per month."
        return nl.join(paras)

    def test_targeted_sends_small_fraction_and_preserves_rest(self):
        src = self.long_doc()
        client = ScriptedClient(edits(("Basic costs $10", "Basic costs $11")))
        doc, st = d.update_document(src, ["Raise the Basic pricing."], client, workers=1)
        self.assertEqual(st.mode, "targeted")
        self.assertEqual(client.calls, 1)
        self.assertLess(st.words_sent, 0.15 * st.words_total)
        self.assertEqual(doc, src.replace("Basic costs $10", "Basic costs $11"))

    def test_sent_words_independent_of_document_size(self):
        sent = []
        for n in (60, 240):
            _, st = d.update_document(self.long_doc(n), ["Raise Basic pricing."], ScriptedClient(
                edits(("Basic costs $10", "Basic costs $11"))), workers=1)
            sent.append(st.words_sent)
        self.assertEqual(sent[0], sent[1])

    def test_global_instruction_sends_everything(self):
        src = self.long_doc(10)
        client = ScriptedClient(*[edits() for _ in range(50)])
        with self.assertRaises(d.UpdateFailed):  # empty edits => no-op failure, but every window was sent
            d.update_document(src, ["Make the tone throughout more formal."], client, workers=1)
        self.assertEqual(client.calls, len(d.chunk_document(src, 1500)))

    def test_no_lexical_hit_falls_back_to_full_scan(self):
        import selector
        src = self.long_doc(10)
        self.assertEqual(selector.select(d.chunk_document(src, 250), ["Zzyzx qqq."])[1], "fallback-full")

    def test_matched_heading_pulls_in_its_section(self):
        import selector
        units = ["## Pricing", "Basic costs $10.", "Pro costs $20.", "## Other", "Unrelated."]
        wanted, _ = selector.select(units, ["Update the pricing section."])
        self.assertTrue(wanted[1] and wanted[2])
        self.assertFalse(wanted[4])


if __name__ == "__main__":
    unittest.main()
