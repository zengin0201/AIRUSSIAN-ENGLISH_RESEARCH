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
from src.agent import ARMS, ARM_LANGUAGE, FEWSHOT, FEWSHOT_TIMES
from src.metrics import fisher_exact
from src.stats import P_EQUALS, P_LESS, format_p


import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "build_tex", pathlib.Path(__file__).resolve().parents[1] / "paper" / "build_tex.py")
_build_tex = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_build_tex)
inline = _build_tex.inline


class TestNormalisation(unittest.TestCase):

    def test_case_and_space_are_ignored(self):
        self.assertEqual(norm("  Lunch  Box "), norm("lunch box"))

    def test_yo_is_folded(self):
        self.assertEqual(norm("зубная щётка"), norm("зубная щетка"))

    def test_punctuation_does_not_break_a_match(self):
        self.assertIn(norm("лекарства"), norm("лекарства$$."))

    def test_time_and_date_survive(self):
        self.assertEqual(norm("07:30"), "07:30")
        self.assertEqual(norm("2026-09-08"), "2026-09-08")

    def test_it_does_not_collapse_two_different_words(self):
        self.assertNotEqual(norm("масло"), norm("мясо"))


class TestParser(unittest.TestCase):

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

    def test_fishers_tea_tasting(self):
        self.assertAlmostEqual(fisher_exact(3, 1, 1, 3), 0.4857, places=4)

    def test_a_significant_table(self):
        self.assertAlmostEqual(fisher_exact(1, 9, 11, 3), 0.002759, places=6)

    def test_it_is_symmetric_under_swapping_the_groups(self):
        self.assertAlmostEqual(fisher_exact(30, 2, 0, 32),
                               fisher_exact(0, 32, 30, 2), places=20)

    def test_no_difference_gives_one(self):
        self.assertAlmostEqual(fisher_exact(5, 5, 5, 5), 1.0, places=10)

    def test_an_empty_margin_is_not_significant(self):
        self.assertEqual(fisher_exact(0, 8, 0, 8), 1.0)

    def test_a_large_table_does_not_lose_precision(self):
        p = fisher_exact(64, 4, 10, 58)
        self.assertLess(p, 1e-21)
        self.assertGreater(p, 1e-23)

    def test_negative_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            fisher_exact(-1, 4, 10, 58)


class TestFewShotBlocks(unittest.TestCase):

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
        from src.agent import SYSTEM_EN
        rendered = (SYSTEM_EN + FEWSHOT["ru"]).format(
            today="2026-09-07", weekday="Monday", tools="(tools)")
        self.assertIn('{"tool": "set_alarm"', rendered)
        self.assertNotIn("{{", rendered)


class TestPValueReporting(unittest.TestCase):

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
        self.assertNotEqual(format_p(0.0114), "p = .01")


class TestExampleTypography(unittest.TestCase):

    def test_double_quotes_become_directional(self):
        self.assertEqual(inline('He said "hello" loudly'),
                         "He said ``hello'' loudly")

    def test_single_quoted_gloss_becomes_directional(self):
        self.assertEqual(inline("means 'twenty past seven' here"),
                         "means `twenty past seven' here")

    def test_a_quotation_is_not_eaten_by_the_gloss_rule(self):
        out = inline('digital: "It is now 19:29, what time in 26 minutes?"')
        self.assertIn("minutes?''", out)
        self.assertNotIn("`'", out)

    def test_apostrophes_inside_words_survive(self):
        for text in ("don't", "the model's output", "Gorman's conventions"):
            self.assertEqual(inline(text), text)

    def test_cyrillic_is_wrapped_without_relying_on_guillemets(self):
        phrase = "двадцать минут"
        out = inline("the phrase " + phrase + " is 07:20")
        self.assertIn("\\ru{" + phrase + "}", out)
        self.assertIn("is 07:20", out)

    def test_paper_body_has_no_quoted_examples(self):
        body = (pathlib.Path(__file__).resolve().parents[1]
                / "paper" / "paper.md").read_text(encoding="utf-8")
        body = body.partition("## Notes to myself")[0]
        self.assertNotIn("«", body, "guillemets are back in the paper body")
        self.assertEqual(len(re.findall('"[^"]+"', body)), 1)


NATURAL_RENDERING = {
    "en": "past",
    "ru": "\u043c\u0438\u043d\u0443\u0442",
    "pl": "po",
    "de": "nach",
}

GERMAN_CLOSED_SET = ("halb", "viertel", "dreiviertel")


class TestFewShotDemonstrations(unittest.TestCase):

    def test_each_block_uses_that_language_ordinary_rendering(self):
        for lang, block in FEWSHOT.items():
            self.assertIn(NATURAL_RENDERING[lang], block.lower(),
                          f"{lang} demonstrations are not in the ordinary form")

    def test_the_rendering_table_covers_every_language(self):
        self.assertEqual(set(FEWSHOT) - set(NATURAL_RENDERING), set())

    def test_german_does_not_demonstrate_its_closed_set(self):
        lowered = FEWSHOT["de"].lower()
        for marker in GERMAN_CLOSED_SET:
            self.assertNotIn(marker, lowered,
                             f"German few-shot demonstrates {marker!r}, "
                             f"which the probe tests directly")

    def test_the_german_probe_does_test_that_closed_set(self):
        with open("data/tasks_time_de.json", "r", encoding="utf-8") as handle:
            tasks = json.load(handle)["tasks"]
        into = [t["instruction"]["de"].lower() for t in tasks
                if t["de_direction"].startswith("into_hour")]
        self.assertTrue(into)
        for wording in into:
            self.assertTrue(any(m in wording for m in GERMAN_CLOSED_SET))


class TestGermanProbe(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open("data/tasks_time_de.json", "r", encoding="utf-8") as handle:
            cls.data = json.load(handle)
        cls.tasks = {t["id"]: t for t in cls.data["tasks"]}

    def test_it_mirrors_the_source_probe_item_for_item(self):
        with open("data/tasks_time_v2.json", "r", encoding="utf-8") as handle:
            source = {t["id"]: t for t in json.load(handle)["tasks"]}
        self.assertEqual(set(self.tasks), set(source))
        for task_id, task in self.tasks.items():
            self.assertEqual(task["instruction"]["en"],
                             source[task_id]["instruction"]["en"],
                             f"{task_id}: English half is not byte-identical")
            self.assertEqual(task["checks"]["de"], source[task_id]["checks"]["en"],
                             f"{task_id}: German is checked against another time")

    def test_both_directions_are_present_in_quantity(self):
        counts = {}
        for task in self.tasks.values():
            counts[task["de_direction"]] = counts.get(task["de_direction"], 0) + 1
        self.assertGreaterEqual(counts.get("into_hour", 0), 5)
        self.assertGreaterEqual(counts.get("from_hour", 0), 8)
        self.assertGreaterEqual(counts.get("digital", 0), 4)

    def test_the_into_hour_cell_is_the_closed_lexical_set(self):
        for task in self.tasks.values():
            if task["de_direction"].startswith("into_hour"):
                words = task["instruction"]["de"].lower()
                self.assertTrue(
                    any(w in words for w in ("halb", "viertel", "dreiviertel")),
                    f"{task['id']}: into_hour without a closed-set word")

    def test_the_from_hour_cell_never_counts_into_the_hour(self):
        for task in self.tasks.values():
            if task["de_direction"] == "from_hour":
                self.assertIn("nach", task["instruction"]["de"].lower(),
                              f"{task['id']}: from_hour without nach")

    def test_the_minimal_pair_sits_at_the_same_clock_position(self):
        quarters = [t for t in self.tasks.values()
                    if t["difficulty"] == "quarter_past"]
        self.assertTrue(all(t["checks"]["de"][0]["time"].endswith(":15")
                            for t in quarters))
        directions = {t["de_direction"] for t in quarters}
        self.assertEqual(directions, {"from_hour", "into_hour_regional"},
                         "the minimal pair needs both readings at :15")

    def test_the_regional_items_are_flagged_so_they_cannot_slip_in(self):
        for task in self.tasks.values():
            self.assertEqual(task["de_regional"],
                             task["de_direction"].endswith("_regional"),
                             f"{task['id']}: flag disagrees with direction")
        flagged = [t["id"] for t in self.tasks.values() if t["de_regional"]]
        self.assertEqual(sorted(flagged), ["g07", "g09"])

    def test_the_note_warns_that_nobody_has_checked_the_german(self):
        self.assertIn("NOT YET CHECKED BY A NATIVE SPEAKER", self.data["meta"]["note"])


if __name__ == "__main__":
    unittest.main()
