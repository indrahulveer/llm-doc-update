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
        doc, _ = d.update_document(src, ["Increase the cost of every plan by 10%."], ScriptedClient(resp), workers=1)
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

    def test_global_instruction_rewrites_paragraphs_and_protects_facts(self):
        nl2 = chr(10) * 2
        src = "# Guide" + nl2 + "Basic costs $10 per month, see https://x.example.com/a for details." + nl2 + "Hey, ya gotta try it!"
        good = json.dumps({"paragraphs": [
            {"id": 2, "text": "Basic costs $10 per month; see https://x.example.com/a for details."},
            {"id": 4, "text": "Please make sure to try it."}]})
        # heading (id 0) is never offered; ids are the paragraph positions
        client = ScriptedClient(good)
        doc, st = d.update_document(src, ["Make the tone throughout more formal."], client, workers=1)
        self.assertIn("Please make sure to try it.", doc)
        self.assertTrue(doc.startswith("# Guide"))
        self.assertEqual(client.calls, 1)

    def test_rewrite_that_drops_a_price_or_url_is_rejected_then_recovers(self):
        nl2 = chr(10) * 2
        src = "# Guide" + nl2 + "Basic costs $10 per month, see https://x.example.com/a."
        bad = json.dumps({"paragraphs": [{"id": 2, "text": "Basic is cheap, see the website."}]})
        good = json.dumps({"paragraphs": [{"id": 2, "text": "Basic costs $10 per month; see https://x.example.com/a."}]})
        client = ScriptedClient(bad, good)
        doc, st = d.update_document(src, ["Make the tone more formal throughout."], client, workers=1)
        self.assertEqual((client.calls, st.retries), (2, 1))
        self.assertIn("; see", doc)

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


NL2 = chr(10) * 2


class GroundingTests(unittest.TestCase):
    def test_all_true_replaces_every_occurrence_and_default_still_rejects_duplicates(self):
        text = "Nimbus Desk is great. Use Nimbus Desk daily."
        out = d.apply_edits(text, [d.Edit("Nimbus Desk", "Harbor Link", 1, True)])
        self.assertEqual(out, "Harbor Link is great. Use Harbor Link daily.")
        with self.assertRaises(d.InvalidEdits):
            d.apply_edits(text, [d.Edit("Nimbus Desk", "Harbor Link", 1, False)])

    def test_quoted_anchor_missing_or_ambiguous_is_skipped_without_a_model_call(self):
        src = "Please contact support. Later, contact support again." + NL2 + "Section A text here."
        self.assertIn("ambiguous", d.check_anchor(src, "Replace the phrase 'contact support' with 'open a ticket'."))
        self.assertIn("not found", d.check_anchor(src, "Remove the section titled 'Data Residency'."))
        self.assertIsNone(d.check_anchor(src, "Replace 'Section A' with 'Section B'."))
        self.assertIsNone(d.check_anchor(src, "Replace every 'contact support' with 'open a ticket'."))
        client = ScriptedClient()
        with self.assertRaises(d.UpdateFailed):          # nothing resolvable -> clean failure, zero calls
            d.update_document(src, ["Replace the phrase 'contact support' with 'x'."], client, workers=1)
        self.assertEqual(client.calls, 0)

    def test_skipped_instruction_is_reported_while_the_rest_is_applied(self):
        src = "Pro costs $20 per month." + NL2 + "Contact support." + NL2 + "Contact support."
        client = ScriptedClient(edits(("Pro costs $20", "Pro costs $22")).replace('{"find"', '{"change": 2, "find"'))
        doc, st = d.update_document(
            src, ["Replace the phrase 'Contact support' with 'Open a ticket'.", "Raise the Pro price by 10%."],
            client, workers=1)
        self.assertIn("$22", doc)
        self.assertEqual([k["instruction"] for k in st.skipped], [1])

    def test_instruction_with_no_attributed_edit_is_reported(self):
        src = "Pro costs $20 per month. Basic costs $10 per month."
        good = json.dumps({"edits": [{"change": 1, "find": "Pro costs $20", "replace": "Pro costs $22"}]})
        doc, st = d.update_document(src, ["Raise Pro by 10%.", "Change the Platinum tier to $999."],
                                    ScriptedClient(good), workers=1)
        self.assertEqual([k["instruction"] for k in st.skipped], [2])

    def test_invalid_change_number_is_rejected_and_retried(self):
        bad = json.dumps({"edits": [{"change": 9, "find": "Pro costs $20", "replace": "x"}]})
        good = json.dumps({"edits": [{"change": 1, "find": "Pro costs $20", "replace": "Pro costs $22"}]})
        client = ScriptedClient(bad, good)
        d.update_document("Pro costs $20 per month.", ["Raise Pro."], client, workers=1)
        self.assertEqual(client.calls, 2)


class BatchTests(unittest.TestCase):
    def test_batch_continues_after_a_failure_and_reports(self):
        import tempfile
        import batch
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            os.makedirs(out)
            good = os.path.join(tmp, "a.json")
            bad = os.path.join(tmp, "b.json")
            with open(good, "w", encoding="utf-8") as f:
                json.dump({"sourceDocument": "Pro costs $20 per month.", "changes": ["Raise Pro."]}, f)
            with open(bad, "w", encoding="utf-8") as f:
                json.dump({"sourceDocument": "Pro costs $20 per month.", "changes": ["Raise Pro."]}, f)
            ok_resp = json.dumps({"edits": [{"change": 1, "find": "Pro costs $20", "replace": "Pro costs $22"}]})
            r1 = batch.run_one("a", good, out, ScriptedClient(ok_resp), True, 1)
            r2 = batch.run_one("b", bad, out, ScriptedClient("x", "x", "x"), True, 1)
            self.assertTrue(r1["ok"] and not r2["ok"])
            self.assertTrue(os.path.exists(os.path.join(out, "a.md")))
            self.assertTrue(os.path.exists(os.path.join(out, "b.fail")))
            self.assertFalse(os.path.exists(os.path.join(out, "b.md")))
            summ = batch.report([r1, r2], out, 1.0)
            self.assertEqual((summ["succeeded"], summ["failed"]), (1, 1))




class RenameAndWideTests(unittest.TestCase):
    def test_literal_rename_needs_no_model_and_covers_derived_forms(self):
        src = "Welcome to Nimbus Desk. Nimbus Desk rocks. curl https://api.nimbusdesk.example.com/v2"
        client = ScriptedClient()
        doc, st = d.update_document(
            src, ["Rename the product from Nimbus Desk to Harbor Link everywhere in the guide."], client, workers=1)
        self.assertEqual(doc, "Welcome to Harbor Link. Harbor Link rocks. curl https://api.harborlink.example.com/v2")
        self.assertEqual(client.calls, 0)
        self.assertEqual(st.changes[-1][0]["occurrences"], 3)

    def test_rename_target_missing_is_reported(self):
        with self.assertRaises(d.UpdateFailed):
            d.update_document("Nothing here.", ["Rename the product from Foo Bar to Baz Qux everywhere."],
                              ScriptedClient(), workers=1)

    def test_duplicate_find_is_rejected_for_targeted_instruction(self):
        src = "If it fails, retry. Later: If it fails, retry."
        dup = json.dumps({"edits": [{"change": 1, "find": "If it fails, retry.", "replace": "x"}]})
        client = ScriptedClient(dup, dup, dup)
        with self.assertRaises(d.UpdateFailed):
            d.update_document(src, ["Reword the retry advice."], client, workers=1)
        self.assertEqual(client.calls, d.MAX_ATTEMPTS)

    def test_unknown_capitalised_target_is_skipped_but_new_names_are_fine(self):
        src = "Basic price $10. Pro price $20."
        self.assertIn("Platinum", d.check_anchor(src, "Change the Platinum tier price to $999 per month."))
        self.assertIsNone(d.check_anchor(src, "Change the Pro price to $25."))
        self.assertIsNone(d.check_anchor(src, "Rename the product from Basic to Zephyr."))

    def test_unrelated_large_deletion_is_rejected(self):
        src = "Intro text about gardening and weather and many other things here." + NL2 + "Pro costs $20."
        big = src.split(chr(10))[0]
        bad = json.dumps({"edits": [{"change": 1, "find": big, "replace": ""}]})
        good = json.dumps({"edits": [{"change": 1, "find": "Pro costs $20", "replace": "Pro costs $22"}]})
        client = ScriptedClient(bad, good)
        d.update_document(src, ["Raise the Pro price by 10%."], client, workers=1)
        self.assertEqual(client.calls, 2)


class PriceAndRecoveryTests(unittest.TestCase):
    PRICING = chr(10).join([
        "## Monthly plans", "- **Basic**: $40 per month.", "- **Pro**: $50 per month.", "## Annual plans",
        "- **Basic**: $400 per year.", "- **Pro**: $500 per year."])

    def run_price(self, instruction):
        out, ch = d.percent_price_change(self.PRICING, instruction)
        return out, ch

    def test_percent_price_scopes_are_exact_and_need_no_model(self):
        out, _ = self.run_price("Raise the monthly plan prices by 10%.")
        self.assertIn("$44 per month", out); self.assertIn("$55 per month", out)
        self.assertIn("$400 per year", out)
        out, _ = self.run_price("Increase only the annual tier prices by 10 percent; leave monthly prices alone.")
        self.assertIn("$440 per year", out); self.assertIn("$40 per month", out)
        out, _ = self.run_price("Raise the Pro price on the annual plan by 10%.")
        self.assertIn("$550 per year", out); self.assertIn("$400 per year", out); self.assertIn("$50 per month", out)
        out, ch = self.run_price("Increase every plan price (monthly and annual) by 10 percent.")
        self.assertEqual(len(ch), 4); self.assertIn("$44 per month", out); self.assertIn("$550 per year", out)
        client = ScriptedClient()
        doc, st = d.update_document(self.PRICING, ["Raise all tier prices by 10%."], client, workers=1)
        self.assertEqual(client.calls, 0)
        self.assertEqual(st.mode, "deterministic")

    def test_edit_starting_inside_heading_marker_is_rejected(self):
        text = "### Legacy API" + chr(10) * 2 + "Old stuff." + chr(10) * 2 + "## Next"
        with self.assertRaises(d.InvalidEdits):
            d.apply_edits(text, [d.Edit("## Legacy API" + chr(10) * 2 + "Old stuff." + chr(10) * 2, "")])

    def test_rewrite_leaves_unfixable_paragraph_unchanged_and_reports_it(self):
        nl2 = chr(10) * 2
        src = "# Guide" + nl2 + "Basic costs $10 per month, see https://x.example.com/a." + nl2 + "Hey, ya gotta try it!"
        bad = json.dumps({"paragraphs": [{"id": 2, "text": "Basic is cheap."},
                                         {"id": 4, "text": "Please make sure to try it."}]})
        still_bad = json.dumps({"paragraphs": [{"id": 2, "text": "Basic is cheap."}]})
        client = ScriptedClient(bad, still_bad, still_bad)
        doc, st = d.update_document(src, ["Make the tone throughout more formal."], client, workers=1)
        self.assertIn("Please make sure to try it.", doc)                       # good paragraph kept
        self.assertIn("Basic costs $10 per month, see https://x.example.com/a.", doc)  # bad one untouched
        self.assertTrue(any("left unchanged" in k["reason"] for k in st.skipped))


if __name__ == "__main__":
    unittest.main()
