"""
Regression tests for the parts of the harness that decide who wins.

Run with the standard library only, no pytest, no GPU:

    python -m tests.run_all

Why these exist. Every number in this project is produced by three pieces of
code: the parser that turns model text into actions, the environment that
executes them, and the verifier that decides success. A bug in any of them
looks exactly like a language effect. The oracle run catches gross breakage;
these tests catch the quiet kind, especially anything that could behave
differently for Cyrillic text than for Latin text.
"""

from __future__ import annotations

import json
import pathlib
import re
import unittest

from src.agent import (
    compact,
    extract_actions,
    protect_values,
    restore_values,
    strip_translation_echo,
)
from src.backends import Generation
from src.environment import Environment, norm, verify
from src.agent import FEWSHOT, FEWSHOT_TIMES
from src.metrics import fisher_exact
from src.stats import P_EQUALS, P_LESS, format_p


class TestNormalisation(unittest.TestCase):
    """norm() is applied to every comparison, so it decides every success."""

    def test_case_and_space_are_ignored(self):
        self.assertEqual(norm("  Lunch  Box "), norm("lunch box"))

    def test_yo_is_folded(self):
        # «щётка» and «щетка» are the same word to a user and must be to us.
        self.assertEqual(norm("зубная щётка"), norm("зубная щетка"))

    def test_punctuation_does_not_break_a_match(self):
        # the translate_safe arm leaves junk like «лекарства$$» glued on
        self.assertIn(norm("лекарства"), norm("лекарства$$."))

    def test_time_and_date_survive(self):
        self.assertEqual(norm("07:30"), "07:30")
        self.assertEqual(norm("2026-09-08"), "2026-09-08")

    def test_it_does_not_collapse_two_different_words(self):
        self.assertNotEqual(norm("масло"), norm("мясо"))


class TestParser(unittest.TestCase):
    """extract_actions turns model text into actions and labels the failure."""

    def test_single_object(self):
        actions, status = extract_actions('{"tool": "stop_music", "args": {}}')
        self.assertEqual(status, "clean")
        self.assertEqual([a["tool"] for a in actions], ["stop_music"])

    def test_array_is_a_batch(self):
        text = ('[{"tool": "set_alarm", "args": {"time": "07:00", "label": "a"}}, '
                '{"tool": "stop_music", "args": {}}]')
        actions, status = extract_actions(text)
        self.assertEqual(status, "clean")
        self.assertEqual(len(actions), 2)

    def test_code_fence_is_recovered(self):
        text = 'Sure!\n```json\n{"tool": "done", "args": {}}\n```'
        actions, status = extract_actions(text)
        self.assertEqual(status, "recovered")
        self.assertEqual(actions[0]["tool"], "done")

    def test_cyrillic_values_survive_the_parser(self):
        text = '{"tool": "create_note", "args": {"text": "взять тетрадь"}}'
        actions, _ = extract_actions(text)
        self.assertEqual(actions[0]["args"]["text"], "взять тетрадь")

    def test_cut_off_batch_yields_nothing(self):
        # This is the exact shape that destroyed the v2 run: a valid plan
        # sliced by the output budget. It must yield ZERO actions, and it
        # must be labelled as a broken structure, not as prose.
        text = ('[{"tool": "add_to_list", "args": {"list_name": "packing", '
                '"item": "паспорт"}}, {"tool": "add_to_list", "args": '
                '{"list_name": "packing", "item": "заря')
        actions, status = extract_actions(text)
        self.assertEqual(actions, [])
        self.assertEqual(status, "malformed_json")

    def test_prose_is_not_confused_with_broken_json(self):
        actions, status = extract_actions("I will set the alarm for you now.")
        self.assertEqual(actions, [])
        self.assertEqual(status, "no_json")

    def test_valid_json_without_a_tool_is_bad_schema(self):
        actions, status = extract_actions('{"thought": "let me think"}')
        self.assertEqual(actions, [])
        self.assertEqual(status, "bad_schema")

    def test_alternative_key_names_are_accepted(self):
        actions, _ = extract_actions('{"name": "play_music", "arguments": {"artist": "Сплин"}}')
        self.assertEqual(actions[0]["tool"], "play_music")
        self.assertEqual(actions[0]["args"]["artist"], "Сплин")

    def test_parser_is_language_blind(self):
        """The same structure must get the same status in both languages."""
        en = '{"tool": "create_note", "args": {"text": "take notebook"}}'
        ru = '{"tool": "create_note", "args": {"text": "взять тетрадь"}}'
        self.assertEqual(extract_actions(en)[1], extract_actions(ru)[1])


class TestTruncationFlag(unittest.TestCase):
    def test_length_means_truncated(self):
        self.assertTrue(Generation("x", 1, 1, 0.0, finish_reason="length").truncated)

    def test_stop_means_not_truncated(self):
        self.assertFalse(Generation("x", 1, 1, 0.0, finish_reason="stop").truncated)

    def test_default_is_not_truncated(self):
        self.assertFalse(Generation("x", 1, 1, 0.0).truncated)


class TestPlaceholderProtection(unittest.TestCase):
    """The ru_translate_safe arm stands or falls on these two functions."""

    def test_roundtrip_restores_every_value(self):
        text = 'Поставь будильник и назови его «автобус», добавь «билет».'
        protected, mapping = protect_values(text)
        self.assertEqual(len(mapping), 2)
        self.assertNotIn("автобус", protected)
        restored, lost, mangled = restore_values(protected, mapping)
        self.assertEqual(restored, text)
        self.assertEqual(lost, 0)
        self.assertEqual(mangled, 0)

    def test_a_dropped_placeholder_is_counted(self):
        _, mapping = protect_values('добавь «билет» и «паспорт»')
        _, lost, _ = restore_values("Add the first item only: __V1__", mapping)
        self.assertEqual(lost, 1)

    def test_damaged_placeholder_is_counted_not_hidden(self):
        # The translator glues junk onto the marker. We must notice, and we
        # must NOT quietly clean it up, because the damage is the finding.
        _, mapping = protect_values('добавь «лекарства»')
        restored, lost, mangled = restore_values("Add __V1__$$ to the list", mapping)
        self.assertEqual(lost, 0)
        self.assertEqual(mangled, 1)
        self.assertIn("$$", restored)

    def test_straight_quotes_are_protected_too(self):
        _, mapping = protect_values("""add 'ticket' and "bus" """)
        self.assertEqual(len(mapping), 2)


class TestTranslationEcho(unittest.TestCase):
    def test_leading_header_is_removed(self):
        self.assertEqual(strip_translation_echo("Request:\nSet an alarm"), "Set an alarm")

    def test_markers_are_removed(self):
        self.assertEqual(
            strip_translation_echo("<<<REQUEST\nSet an alarm\nREQUEST>>>"),
            "Set an alarm",
        )

    def test_clean_translation_is_untouched(self):
        text = 'Set an alarm and call it «автобус».'
        self.assertEqual(strip_translation_echo(text), text)


class TestEnvironment(unittest.TestCase):
    def test_alarm_roundtrip(self):
        env = Environment()
        obs, status = env.call("set_alarm", {"time": "07:30", "label": "подъём"})
        self.assertEqual(status, "ok")
        self.assertEqual(env.state["alarms"][0]["label"], "подъём")

    def test_bad_time_is_rejected_the_same_in_any_language(self):
        env = Environment()
        for bad in ("7.30", "half past seven", "полвосьмого", "25:00"):
            _, status = env.call("set_alarm", {"time": bad, "label": "x"})
            self.assertEqual(status, "bad_args", f"{bad!r} should be rejected")

    def test_unknown_tool_is_reported(self):
        _, status = Environment().call("send_email", {"to": "x"})
        self.assertEqual(status, "unknown_tool")

    def test_list_name_is_normalised_but_item_is_not(self):
        env = Environment()
        env.call("add_to_list", {"list_name": "Shopping", "item": "Молоко"})
        self.assertIn("shopping", env.state["lists"])
        self.assertEqual(env.state["lists"]["shopping"], ["Молоко"])

    def test_removing_a_missing_item_fails_loudly(self):
        env = Environment()
        env.call("add_to_list", {"list_name": "shopping", "item": "хлеб"})
        _, status = env.call("remove_from_list", {"list_name": "shopping", "item": "сыр"})
        self.assertEqual(status, "bad_args")

    def test_extra_arguments_do_not_crash_the_tool(self):
        _, status = Environment().call(
            "set_alarm", {"time": "07:30", "label": "x", "repeat": "daily"}
        )
        self.assertEqual(status, "ok")


class TestVerifier(unittest.TestCase):
    """The verifier decides success. It must never favour one language."""

    def test_empty_and_wrong_value_are_distinguished(self):
        checks = [{"kind": "alarm_exists", "time": "05:45", "label_contains": "bus"}]

        nothing = verify({"alarms": [], "calendar": [], "lists": {},
                          "music": {"playing": None}, "notes": []}, checks)
        self.assertFalse(nothing["success"])
        self.assertEqual(nothing["failure_kind"], "empty")

        wrong = verify({"alarms": [{"time": "06:00", "label": "bus"}], "calendar": [],
                        "lists": {}, "music": {"playing": None}, "notes": []}, checks)
        self.assertFalse(wrong["success"])
        self.assertEqual(wrong["failure_kind"], "wrong_value")

    def test_partial_credit_is_counted_but_is_not_success(self):
        state = {"alarms": [{"time": "07:00", "label": "подъём"}], "calendar": [],
                 "lists": {}, "music": {"playing": None}, "notes": []}
        result = verify(state, [
            {"kind": "alarm_exists", "time": "07:00", "label_contains": "подъём"},
            {"kind": "alarm_exists", "time": "22:30", "label_contains": "спать"},
        ])
        self.assertFalse(result["success"])
        self.assertEqual((result["passed"], result["total"]), (1, 2))

    def test_duplicate_actions_do_not_break_a_contains_check(self):
        # models often re-emit a whole batch; that wastes tokens but the
        # state is still correct, and the verifier must agree
        state = {"alarms": [], "calendar": [], "lists": {"packing": ["носки", "носки"]},
                 "music": {"playing": None}, "notes": []}
        result = verify(state, [{"kind": "list_contains", "list_name": "packing",
                                 "item": "носки"}])
        self.assertTrue(result["success"])

    def test_absence_checks_work(self):
        state = {"alarms": [], "calendar": [], "lists": {"shopping": ["чай"]},
                 "music": {"playing": None}, "notes": []}
        result = verify(state, [
            {"kind": "alarm_absent", "time": "06:00"},
            {"kind": "music_stopped"},
            {"kind": "list_not_contains", "list_name": "shopping", "item": "сыр"},
        ])
        self.assertTrue(result["success"])

    def test_the_two_languages_are_scored_by_the_same_rule(self):
        """Same shape, same words in either language, same verdict."""
        en_state = {"alarms": [{"time": "07:20", "label": "school"}], "calendar": [],
                    "lists": {}, "music": {"playing": None}, "notes": []}
        ru_state = {"alarms": [{"time": "07:20", "label": "школа"}], "calendar": [],
                    "lists": {}, "music": {"playing": None}, "notes": []}
        en = verify(en_state, [{"kind": "alarm_exists", "time": "07:20",
                                "label_contains": "school"}])
        ru = verify(ru_state, [{"kind": "alarm_exists", "time": "07:20",
                                "label_contains": "школа"}])
        self.assertEqual(en["success"], ru["success"])
        self.assertEqual(en["passed"], ru["passed"])


class _FakeBackend:
    def count_tokens(self, text):
        return len(text.split())


class TestCompaction(unittest.TestCase):
    def test_no_budget_means_no_change(self):
        messages = [{"role": "system", "content": "a b c"}] * 8
        out, dropped = compact(messages, _FakeBackend(), None)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(out), 8)

    def test_over_budget_drops_the_oldest_turn(self):
        messages = [{"role": "system", "content": "word " * 20}] * 8
        out, dropped = compact(messages, _FakeBackend(), 60)
        self.assertGreater(dropped, 0)
        self.assertLess(len(out), 8)

    def test_the_system_prompt_and_the_request_are_never_dropped(self):
        messages = [{"role": "system", "content": "SYSTEM " * 50},
                    {"role": "user", "content": "REQUEST " * 50}]
        messages += [{"role": "assistant", "content": "x " * 50}] * 6
        out, _ = compact(messages, _FakeBackend(), 10)
        self.assertIn("SYSTEM", out[0]["content"])
        self.assertIn("REQUEST", out[1]["content"])


class TestFisherExact(unittest.TestCase):
    """
    Every significance claim in the paper comes from this function.

    The p-values in earlier drafts were not produced by any code in this
    repository and were wrong by up to nine orders of magnitude (F32). These
    tests pin the function to published worked examples so the same thing
    cannot happen silently again.
    """

    def test_fishers_tea_tasting(self):
        # Fisher's own example. The textbook answer is 0.4857.
        self.assertAlmostEqual(fisher_exact(3, 1, 1, 3), 0.4857, places=4)

    def test_a_significant_table(self):
        # Worked example reproduced by R's fisher.test: p = 0.002759.
        self.assertAlmostEqual(fisher_exact(1, 9, 11, 3), 0.002759, places=6)

    def test_it_is_symmetric_under_swapping_the_groups(self):
        # Which language is the first row must not change the p-value.
        self.assertAlmostEqual(fisher_exact(30, 2, 0, 32),
                               fisher_exact(0, 32, 30, 2), places=20)

    def test_no_difference_gives_one(self):
        self.assertAlmostEqual(fisher_exact(5, 5, 5, 5), 1.0, places=10)

    def test_an_empty_margin_is_not_significant(self):
        # Nobody succeeded anywhere: there is no evidence of anything.
        self.assertEqual(fisher_exact(0, 8, 0, 8), 1.0)

    def test_a_large_table_does_not_lose_precision(self):
        # The main result: English 64/68 against Russian 10/68. This is the
        # value the earlier drafts reported as 2.6e-13.
        p = fisher_exact(64, 4, 10, 58)
        self.assertLess(p, 1e-21)
        self.assertGreater(p, 1e-23)

    def test_negative_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            fisher_exact(-1, 4, 10, 58)


class TestFewShotBlocks(unittest.TestCase):
    """
    The few-shot demonstrations live in the prompt, not in a task file, so
    `validate_tasks.py` cannot see them. Nothing else checks that the English
    arm and the Russian arm are shown the same thing.

    An English block with easier or more numerous demonstrations would
    reproduce the paper's entire effect without any model behaving oddly, and
    no existing guard would notice. These tests are that guard.
    """

    def _times_in(self, text):
        return re.findall(r'"time": "(\d\d:\d\d)"', text)

    def test_every_language_demonstrates_the_same_clock_times(self):
        for lang, block in FEWSHOT.items():
            self.assertEqual(self._times_in(block), list(FEWSHOT_TIMES),
                             f"{lang} demonstrates different times")

    def test_every_language_shows_the_same_number_of_examples(self):
        counts = {lang: block.count("User:") for lang, block in FEWSHOT.items()}
        self.assertEqual(len(set(counts.values())), 1, counts)
        for lang, block in FEWSHOT.items():
            self.assertEqual(block.count("User:"), block.count("You:"), lang)

    def test_no_demonstrated_time_appears_in_any_probe(self):
        # A demonstration the model is later tested on would measure recall of
        # three items, not acquisition of a rule.
        used = set()
        for path in sorted(pathlib.Path("data").glob("tasks_*.json")):
            with open(path, "r", encoding="utf-8") as handle:
                for task in json.load(handle)["tasks"]:
                    for check in task["checks"]["en"]:
                        if "time" in check:
                            used.add(check["time"])
        overlap = used & set(FEWSHOT_TIMES)
        self.assertFalse(overlap, f"demonstrated time also in a probe: {overlap}")

    def test_the_block_survives_prompt_formatting(self):
        # SYSTEM_EN goes through str.format, so a single brace in the examples
        # would raise instead of rendering the JSON.
        from src.agent import SYSTEM_EN
        rendered = (SYSTEM_EN + FEWSHOT["ru"]).format(
            today="2026-09-07", weekday="Monday", tools="(tools)")
        self.assertIn('{"tool": "set_alarm"', rendered)
        self.assertNotIn("{{", rendered)


class TestPValueReporting(unittest.TestCase):
    """
    Gorman's convention, and the regex that polices it.

    The first version of that regex matched ".024." at the end of a sentence
    and crashed on float(), so `--check` stopped checking the moment the paper
    adopted the convention the checker exists to enforce. It failed silently:
    the header printed, nothing followed, and the exit code looked like
    ordinary violations.
    """

    def test_a_trailing_sentence_period_is_not_part_of_the_number(self):
        self.assertEqual(P_EQUALS.findall("the result (p = .024)."), [".024"])
        self.assertEqual(P_EQUALS.findall("p = .024."), [".024"])
        self.assertEqual(P_LESS.findall("p < .01."), [".01"])

    def test_every_match_converts_to_a_float(self):
        text = "p = .024. p = 1.4e-22, p = 0.47; p = 1.0 and p < .01."
        for raw in P_EQUALS.findall(text) + P_LESS.findall(text):
            float(raw if not raw.startswith(".") else "0" + raw)

    def test_below_alpha_reports_the_threshold(self):
        for tiny in (1.36e-22, 6.1e-16, 0.00125, 0.0099):
            self.assertEqual(format_p(tiny), "p < .01")

    def test_above_alpha_reports_two_significant_digits(self):
        self.assertEqual(format_p(0.47), "p = .47")
        self.assertEqual(format_p(0.0114), "p = .011")
        self.assertEqual(format_p(1.0), "p = 1.0")

    def test_a_value_just_above_alpha_is_not_printed_as_alpha(self):
        # At two decimal places .0114 prints as ".01", which reads as if the
        # test landed exactly on the threshold.
        self.assertNotEqual(format_p(0.0114), "p = .01")


if __name__ == "__main__":
    unittest.main()
