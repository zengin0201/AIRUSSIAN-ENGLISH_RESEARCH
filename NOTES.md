# Lab notebook

Every run, every number, every decision. Written in English because most of it
goes straight into the paper (Method and Limitations) and a reviewer may read it.

Rule: write the entry on the same day. A number without its context is useless
six months later.

Hardware: RTX 2060 6GB, i5-10400F, 16GB RAM, Windows.
Stack: torch 2.13.0+cu126, transformers, bitsandbytes 4-bit.
GPU idle draw: 27.5-39.7 W (it moves, so measure it at the start of every run).

---

## 2026-08-18 — session 1

### Runs

| file | model | tasks | arms | repeats |
|---|---|---|---|---|
| `oracle.jsonl` | none (gold replay) | 12 | en, ru | 3 |
| `smoke.jsonl` | Qwen2.5-1.5B | 3 | en, ru | 1 |
| `pilot.jsonl` | Qwen2.5-3B | 12 | all 4 | 1 |
| `safe.jsonl` | Qwen2.5-3B | 12 | ru, translate, safe | 1 |
| `safe2.jsonl` | Qwen2.5-3B | 12 | ru, translate, safe | 1 |

Oracle passed 72/72 on every version of the code. Re-run it after any change
to tasks or to the environment.

### pilot.jsonl — main numbers

Static fertility RU/EN = 1.40 (measured with the Qwen2.5-3B tokenizer).

| arm | success | steps | tok/ep | CPST | parse_err | tool_err |
|---|---|---|---|---|---|---|
| en | 75.0% | 3.67 | 2530.7 | 3374.2 | 0.833 | 0.417 |
| ru | 66.7% | 3.42 | 2427.2 | 3640.8 | 0.250 | 0.667 |
| ru_sys_ru | 75.0% | 3.08 | 2489.8 | 3319.8 | 0.250 | 0.750 |
| ru_translate | 8.3% | 3.50 | 2610.1 | 31321.0 | 0.417 | 0.333 |

Decomposition RU/EN: step 0.932 x density 1.029 x success 1.125 = 1.079.
Measured CPST ratio 1.079. Bootstrap 95% CI **[0.532, 2.099]**.

**Reading: no effect detected.** The interval covers 1.0 with a lot of room.
At 12 tasks and one repeat the study cannot see anything. This number must not
be quoted anywhere until the task set is 60 and repeats are 3.

The main hypothesis (agentic gap larger than the tokenizer ratio) is *not*
supported so far: 1.079 measured against 1.40 predicted, i.e. the wrong
direction.

### safe2.jsonl — after the identifier fix

Fertility 1.395.

| arm | success | steps | tok/ep | CPST | parse_err |
|---|---|---|---|---|---|
| ru | 66.7% (8/12) | 3.50 | 2491.6 | 3737.4 | 0.333 |
| ru_translate | 8.3% (1/12) | 4.17 | 3136.0 | 37632.0 | 0.833 |
| ru_translate_safe | 58.3% (7/12) | 3.50 | 2599.7 | 4456.6 | 0.417 |

Mean translation overhead: 180.5 tokens per episode.

Before the identifier fix the safe arm was 6/12; after it, 7/12. So the
artifact was worth exactly one task.

**Reading:**
1. Naive translate-then-act is catastrophic (1/12 against 8/12). This gap is
   large enough to trust even at n=12.
2. Placeholder protection repairs almost all of it (7/12 against 8/12). The
   remaining difference is one task, which is nothing at this sample size.
3. But the policy still does not pay: CPST 4456 against 3737, about 19% more
   per successful task, for no benefit. On tasks of 2-4 steps, translating
   first is a loss even when done correctly.

Open: find the break-even. Needs tasks of 6-8 steps.

---

## Findings

### F1 — models batch several tool calls in one turn
Both `[{...},{...}]` and two objects side by side. The first parser took one
action and threw the rest away, then the model called `done` believing the
work was finished. Two of six smoke failures came from this.

**Changed:** the parser now collects every action in the output and executes
them in order. The system prompt allows an array explicitly, identically in
both languages. Logged as `n_actions` and `batched_turns`.

**Why it had to change:** rejecting valid JSON because of its arrangement
measures the strictness of the harness, not the language. If one language
batches more than the other, the artifact would look like an effect.

### F2 — models sometimes emit broken JSON syntax
Seen: `[["add_event", ["date": "2026-09-11", ...]]]`. Square brackets where
braces belong. Not parseable.

**Not changed. Deliberately.** A repair heuristic cannot be falsified, and
this particular failure appeared in the English arm, so repairing it would
improve the English side. Any later reviewer would be right to distrust the
whole result.

**Changed instead:** a separate status `malformed_json` (tried to produce a
structure, broke the syntax) against `no_json` (answered in prose, never
tried). The first is the interesting one for the schema-mismatch hypothesis.

> **Superseded by F11 (session 3).** Most of what this status counted was not
> the model breaking syntax at all. It was our own output budget cutting a
> correct batch in half. See F11 before using any `malformed_json` number.

### F3 — episodes ran to the step ceiling
The model finished the work and then never called `done`, burning the whole
budget. With both arms saturating at the same ceiling, `step_factor` is
forced to 1.0 and the measurement dies.

**Changed:** two stop rules, identical for all arms, logged in
`stopped_reason`. (a) state unchanged for 3 turns -> `stalled`. (b) two turns
in a row containing only already-executed actions -> `looping`. Also
`max_steps` 12 -> 8 (the longest gold trace is 5) and `max_new_tokens`
200 -> 160.

Rule (a) alone was not enough: a model re-adding the same item does change the
state each time. Rule (b) catches that.

> **Note added in session 3.** The `max_new_tokens` 200 -> 160 change made here
> is what caused F11. The two decisions were bundled into one commit and the
> second one was never tested on its own.

### F4 — naive translation destroys user content
The translation prompt asks explicitly to keep the user's words in Russian.
The 3B model ignores it.

Examples: «спортзал» -> `gym`, «протеиновый батончик» -> `protein bar`,
«поздний старт» -> `late start`. All 11 failures were `wrong_value`: the agent
did the work but wrote the wrong words into the user's data.

Preservation is inconsistent, not absent: in t03 «Кино» survived but
«наушники» became `earphones`. Proper nouns survive, common nouns do not.

**Changed:** new arm `ru_translate_safe`. Quoted content is replaced by
`__V1__`, `__V2__` placeholders before translation and restored after. The
content is simply not present in the text the translator sees, so it cannot be
damaged. Logged as `placeholders_total` and `placeholders_lost`.

**Careful with the claim.** This does not show that translate-then-act is a
bad policy. It shows that a *naive* implementation with a *small* model loses
content even when told not to. A larger model might comply.

### F5 — my own artifact: the list identifier was translated
`в список shopping` -> `to the shopping list` -> the model created a second
list literally named `shopping list`, while the checks look for `shopping`.

The README claims list names are English identifiers held constant across
arms, as a control. The translation arm broke exactly that control, so part of
its failures measured a flaw in my task text.

**Changed:** list names are now quoted in both languages, so placeholder
protection covers them too. Worth one task out of twelve.

### F6 — translation silently corrupted a time
t08. Original: «полвосьмого утра» = 07:30. Translation: "half past six in the
morning" = 06:30.

Russian «полвосьмого» means *half before eight*; English "half past" means
*half after*. The model fell into the inversion.

Placeholders cannot help here: this is not quoted content, it is meaning
carried by the sentence itself. A translate-then-act pipeline can move an
alarm by an hour without any error being raised anywhere.

**Not fixed. This is a result, not a bug.** Next step is a dedicated subset:
spoken times, relative dates, numerals. Ten tasks like this would show the
effect cleanly instead of once by accident.

### F7 — the English arm produced the most schema failures
`bad_schema`: en 10, ru 3, ru_sys_ru 3, ru_translate 5.

This is the opposite of the schema-mismatch hypothesis. No explanation yet.
May well be one task behaving badly. Check again at n=60.

### F8 — a Russian system prompt did not hurt
`ru_sys_ru` reached 75%, equal to English and above `ru` at 66.7%. One task of
difference, so it means nothing yet, but if it survives at n=60 it is a
concrete and non-obvious recommendation for builders.

---

## Open questions

1. Does any RU/EN gap survive at 60 tasks and 3 repeats, or is the null real?
2. Where is the break-even for translate-then-act? Needs 6-8 step tasks.
3. Is F7 (English schema failures) real or one bad task?
4. Does F8 (Russian system prompt) hold up?
5. Does the gap shrink on a large model? Free API tier can answer this at zero
   cost.
6. Joules per successful task: energy was logged but not yet analysed.

## Next actions

- [ ] Expand 12 -> 60 tasks using MASSIVE (parallel, human-translated, CC-BY-4.0)
- [ ] Add a numerals/time/date subset for F6
- [ ] Add tasks of 6-8 steps for question 2
- [ ] Full run: 2 local models, 3 repeats, temperature 0.7
- [ ] Robustness check on a free API tier model
- [ ] Analyse the energy logs

---

## 2026-08-18 — session 2

### New task set: `data/tasks_v2.json` (14 tasks)

The project pivots. The original hypothesis (agentic gap larger than the
tokenizer ratio) looks null at n=12, while the translate-then-act line keeps
producing clean results. So the next task set is not a general expansion. It
targets the two open questions that are actually alive.

**Group A (a01-a08) — semantic traps.** Time, date and numeral expressions
that translate badly. Placeholder protection cannot help here, because the
meaning sits in the sentence, not in a quoted value. Built from F6.

| id | trap | correct answer |
|---|---|---|
| a01 | «без четверти шесть» | 05:45 |
| a02 | «в половине третьего» | 14:30 |
| a03 | «двадцать минут восьмого» | 07:20 |
| a04 | «пять минут девятого вечера» | 20:05 |
| a05 | «послезавтра» | 2026-09-09 |
| a06 | «во вторник на следующей неделе» | 2026-09-15 |
| a07 | «за полтора часа до» | 19:00 minus 90 min = 17:30 |
| a08 | «через сутки» | +24 h, next day same time |

Russian counts *toward* the coming hour («полвосьмого» = half before eight =
07:30); English counts *from* the past hour ("half past seven" = 07:30 too,
but "half past six" = 06:30). The inversion is exactly what broke t08.

**Group B (b01-b06) — long chains**, 6 to 9 actions. Translation overhead is
fixed at about 180 tokens per episode, so if translate-then-act ever pays off,
it pays off on long tasks. This group is built to find that crossing point.

### F9 — the step budget punished long tasks
Oracle failed b03 and b06: 9-10 gold actions against a fixed budget of 8.
The ceiling, not the difficulty, decided the outcome.

**Changed:** the budget is now `max(--max-steps, n_gold_actions + 3)`,
computed per task, identical for every arm, logged as `step_budget`.

Oracle after the fix: 84/84 on v2, 72/72 on seed.

### Next run

    python -m src.run_experiment --backend local --model Qwen/Qwen2.5-3B-Instruct \
        --tasks data/tasks_v2.json --arms ru ru_translate ru_translate_safe \
        --repeats 1 --out results/v2.jsonl

Then compare group A against group B separately. Group A answers "does
translation corrupt meaning". Group B answers "where is the break-even".

### v2.jsonl — first run on the targeted set

Model Qwen2.5-3B, 14 tasks, 1 repeat. Static fertility 1.38.

| arm | success | steps | tok/ep | CPST | parse_err |
|---|---|---|---|---|---|
| ru | 7.1% (1/14) | 3.86 | 3209.9 | 44939.0 | 0.714 |
| ru_translate | 7.1% (1/14) | 3.43 | 2909.7 | 40736.0 | 0.500 |
| ru_translate_safe | 7.1% (1/14) | 4.50 | 4144.3 | 58020.0 | 1.429 |

Successes: a05 in `ru`, b02 in `ru_translate`, a08 in `ru_translate_safe`.
Mean translation overhead 224.9 tokens per episode.

**Floor effect. Nothing here is interpretable.** The line printed by
`analyze` — `saving per successful task = 4203 tokens -> WORTH IT` — is an
artifact of dividing by a single success. Do not quote it anywhere.

### F10 — a diagnostic probe must not be mixed with general capability
Group A tasks ask the model to resolve a time trap *and* add an item to a
list. When the second part fails, the first part is unobservable. Binary
success hides exactly the thing the group was built to measure.

**To change:** group A tasks become single-action (set the alarm, nothing
else). Also read `checks_passed` per check, not only the boolean.

Group B has a different problem: 3B cannot hold a 6-9 action chain at all.
That one needs a stronger model from a free tier, not a task redesign.

> **The last paragraph is wrong. Corrected by F11 below.** The 3B model was
> never given the chance to hold the chain. Do not spend time on a free-tier
> API for this reason.

---

## 2026-08-18 — session 3

Nothing was run on the GPU in this session. It was spent reading the v2 traces
turn by turn, and the whole v2 result turned out to be a harness artifact for
the fifth time.

### F11 — `malformed_json` was our output budget, not the model

In `results/v2.jsonl`, across 165 turns:

| parse_status | count | of those, hit `max_new_tokens=160` |
|---|---|---|
| clean | 104 | 0 |
| recovered | 24 | 4 |
| malformed_json | 32 | **32** |
| bad_schema | 4 | 0 |
| no_json | 1 | 0 |

**Every single `malformed_json` turn was a generation that ran out of output
budget. Not one occurred below the cap.** The model was not breaking JSON. It
was writing a correct batch of 6-9 actions and we were slicing it mid-string.

This matters more than a mislabelled counter, because of how the loop then
behaves. A cut batch parses to **zero** actions, not to a partial plan. The
agent gets "your answer contained no valid tool call", re-sends the same
too-long batch, and dies on the loop detector. One truncated turn is usually
the whole episode.

Episodes containing at least one cut turn: 1 success out of 18.
Episodes with no cut turn: 2 out of 24.

### F11a — group B was never tested for capability

Turn 0 of every group-B episode in v2:

| task | arm | turn-0 completion tokens | outcome |
|---|---|---|---|
| b01 | ru | 160 — cut | 0 actions |
| b01 | ru_translate | **159 — fit** | **6 actions, all correct** |
| b01 | ru_translate_safe | 160 — cut | 0 actions |
| b02 | ru_translate | 160 — cut, 3 recovered | went on to pass 7/7 |
| ... | | | |

17 of 18 group-B opening turns were cut at the cap. The one that fit — by a
single token, because its content was English — emitted all six actions
correctly on the first attempt. The one that recovered three actions from a
cut turn went on to solve the whole task.

So the claim in F10 that "3B cannot hold a 6-9 action chain" has no support in
this data. The model held the chain the one time we let it finish the sentence.

**b01 is the cleanest illustration in the project so far:** same task, same
model, same step budget. English content fits in 159 tokens and succeeds
mechanically. Russian content needs more than 160 for the identical plan and
produces nothing at all.

### The cliff is the finding, not the bug

The naive fix is to raise `max_new_tokens` and move on. That would throw away
the most interesting thing in the run.

An agent answers with a batch of tool calls. The output budget is a hard edge:
one token over and the turn yields nothing. The same plan costs about 1.4x
more output tokens in Russian. So a single output budget, applied to every
language the way every deployment applies it, gives non-English users less
room for the same work. This is not a smooth tax proportional to fertility —
it is a step function, and nobody has measured where the step is.

**Planned:** sweep `max_new_tokens` over 96/128/192/288/384 on group B, arms
`en` and `ru`, and read off the horizontal distance between the two success
curves. `src/analyze_sweep.py` does this. If that distance is near 1.4x, the
tokenizer tax shows up in a place nobody looked, with a direct practical
consequence.

### F12 — the translate prompts leaked their own header

`ru_translate_safe` instructions came out beginning with `Request:` because
the prompt ended with `Request:\n{instruction}` and the model copied the
label. That is our scaffolding entering the measured instruction.

Also seen: placeholders returning with junk glued on, `__V6__$$` restoring to
`лекарства$$.`.

**Changed:** the request now sits between explicit markers, and
`strip_translation_echo()` removes any copied header. Damaged placeholders are
**counted, not cleaned** (`placeholders_mangled`) — the damage is the thing the
arm exists to measure, so hiding it would defeat the arm.

### Harness changes in this session

- `backends.py`: `Generation.finish_reason` and `.truncated`. The local backend
  decides by looking for a stop token in the tail; the API backend reads
  `finish_reason` from the response.
- `agent.py`: statuses `truncated` and `truncated_partial`, separate from
  `malformed_json`, so F2's distinction finally means what it claims.
- `agent.py`: on a cut answer the agent is told so and asked to send at most
  two actions. Identical English sentence in every arm, and switchable with
  `--no-truncation-hint` so its value can be measured rather than assumed.
  This is feedback from the environment, like "unknown tool" — it is not
  repair of the model's output, so F2's rule still holds.
- `metrics.py` / `analyze.py`: `rate_any_truncation`, `rate_looping`, success
  broken down by `difficulty`, placeholder health, and `scaffold_share()`.
- `run_experiment.py`: `--seed`, `--no-truncation-hint`, and `max_new_tokens`
  recorded in the meta line so a sweep can be analysed afterwards.

### New tooling

- **`src/validate_tasks.py`.** Checks statically that the English and Russian
  halves of a task are the same task: same checks in the same order, same
  times, dates, list names and sizes, same sequence of tools in the gold
  trace. Only the human-readable words may differ. This is the one guard
  against the hidden-asymmetry failure, and until now nothing checked it —
  the oracle proves the gold trace satisfies the checks, which is a different
  claim. Both task files pass.
- **`tests/`** — 38 tests, standard library only, no pytest, no GPU, ~2
  seconds. They cover the parser (including the exact cut-batch shape from
  F11), the verifier, `norm()` on Cyrillic, placeholder round-trips, and
  compaction. Two of them assert that the parser and the verifier behave
  identically on Latin and Cyrillic input.
- **`requirements.txt`** — did not exist at all, although the README tells you
  to install from it. Reproducing the project failed at step one.
- `python -m tests.run_all` runs everything and validates every task file.

Oracle after all changes: 42/42 on `tasks_v2.json`.

### Reading of session 3

`results/v2.jsonl` should be treated as a harness measurement, not a language
measurement. Do not quote any number from it. The floor at 7.1% is explained
without any appeal to model capability or to language difficulty.

This is the fifth time in this project that an apparent effect was our own
bug (batching, the step ceiling, the list identifier, the step budget for long
tasks, and now the output budget). The rule in HANDOFF §12 keeps paying for
itself.

### Still open after session 3

1. **The main comparison has never been run on `tasks_v2.json`.** There is no
   `en` arm in `v2.jsonl`. Until that run exists, the pivot to group A/B has
   produced no EN/RU number at all.
2. F10's first half stands: group A conflates the trap with a second action,
   and needs a single-action redesign so the probe is clean.
3. Where the output-budget cliff sits for each language.

---

## 2026-08-18 — session 4

### v3_main.jsonl — the first run with an `en` arm on the targeted set

Qwen2.5-3B, `tasks_v2.json`, 14 tasks, 1 repeat, `--max-new-tokens 384`.
Static fertility 1.38.

| arm | success | steps | tok/ep | CPST | cut% | loop% |
|---|---|---|---|---|---|---|
| en | 57.1% (8/14) | 3.57 | 2911 | 5094 | 0 | 71 |
| ru | 28.6% (4/14) | 3.64 | 3167 | 11083 | 0 | 50 |
| ru_sys_ru | 35.7% (5/14) | 3.07 | 2913 | 8157 | 0 | 50 |
| ru_translate_safe | 35.7% (5/14) | 4.79 | 4305 | 12055 | 0 | 71 |

Decomposition RU/EN: step 1.020 x density 1.066 x success 2.000 = 2.176.
Measured CPST ratio 2.176. Bootstrap 95% CI **[0.754, 7.964]**.

**F11 confirmed by the fix.** `cut%` is 0.0 in every arm at a 384-token budget,
against 43% of episodes at 160. Group B went from 0/6 in every arm to en 4/6,
ru 3/6, ru_sys_ru 3/6, safe 4/6. The 3B model holds a 6-9 action chain
perfectly well. F10's closing claim is now definitively dead, and the free-tier
API run planned for that reason is cancelled.

**On the headline: it points the predicted way for the first time (2.18 against
1.38 predicted, 1.58x more than the tokenizer accounts for) and it still cannot
be quoted.** The interval [0.75, 7.96] covers 1.0. Fourteen tasks, one repeat.
The direction is now worth chasing; the number is not a result.

The shape of the decomposition is the interesting part and it is stable with
what session 1 found: `step` 1.02 and `density` 1.07 are both near nothing,
`success` is 2.0. Whatever the gap is, it is not tokens and it is not steps.
It is that the Russian episode fails.

### F13 — Russian spoken time is read exactly one hour late

This is the cleanest thing the project has produced. Group A, the four tasks
whose trap is the Russian construction that names the hour being counted
*into*, with the first time each arm actually wrote:

| task | correct | en | ru | ru_sys_ru | ru_translate_safe |
|---|---|---|---|---|---|
| a02 «половина третьего дня» | 14:30 | 14:30 (+0) | 15:30 (**+60**) | 13:30 (**−60**) | 15:30 (**+60**) |
| a03 «двадцать минут восьмого» | 07:20 | 07:20 (+0) | 08:20 (**+60**) | 08:20 (**+60**) | 08:20 (**+60**) |
| a04 «пять минут девятого вечера» | 20:05 | 20:08 (+3) | 21:05 (**+60**) | 21:05 (**+60**) | 21:55 (+110) |
| a07 «за полтора часа до 19:00» | 17:30 | 17:30 (+0) | 18:30 (**+60**) | 18:30 (**+60**) | 18:30 (**+60**) |

English wrote the correct time in three of four. Every Russian arm wrote it in
zero of four. **Ten of the twelve Russian answers are wrong by exactly +60
minutes; an eleventh is wrong by exactly −60.** One outlier.

The mechanism is a grammatical difference, not a capability gap:

* Russian «двадцать минут восьмого» = *twenty minutes of the eighth [hour]*.
  The named ordinal is the hour being counted **into**, so this is 07:20.
* English "twenty past seven" names the hour already **completed**.
* The model treats the Russian ordinal as the clock hour and lands one hour
  late, every time.

`a07` is the same family with a different trigger: «полтора часа» is 90
minutes, and every Russian arm subtracted 30, reading «пол-» as "half" and
dropping the "one and a".

**Three interventions were tested and none of them help.** A Russian system
prompt (`ru_sys_ru`) makes the same +60 error. Translation with placeholder
protection (`ru_translate_safe`) makes the same +60 error, because the meaning
lives in the sentence and not in a quoted span — this is F6 generalised from
one accidental observation to a systematic one.

Why this matters more than the CPST number:

1. It is **silent**. The alarm is set, the tool returns success, no error is
   raised anywhere. The user finds out by missing the bus.
2. It is **invisible to every cost metric**. Tokens, steps and fertility say
   nothing about it.
3. It is **deterministic**, not a rate. The same error appears in three
   independent arms of the same run.
4. It **generalises by construction**. Any language whose spoken clock counts
   into the coming hour is exposed the same way — German "halb acht" is 07:30,
   and Czech, Polish, Danish, Swedish and Dutch share the pattern. The claim to
   test later is that this is a property of the construction, not of Russian.

**Caveat, stated plainly:** n = 1 repeat, 4 tasks, so this is a pilot signal.
The consistency across three arms is what makes it worth a dedicated probe, not
the sample size.

### `data/tasks_time.json` — the probe built from F13

18 single-action tasks, one `set_alarm` call and one check on the time, so the
trap can never be hidden behind an unrelated failure. This is F10's first half,
finally done. Four categories:

| category | n | what it isolates |
|---|---|---|
| `into_hour` | 7 | the F13 construction |
| `subtractive` | 3 | «без N» / "N to" — hard in *both* languages in v3_main, so it separates "Russian time is hard" from "this construction is hard" |
| `literal` | 5 | digital times, identical in both languages; must score near 100% or the probe is broken |
| `arithmetic` | 3 | relative offsets, including «полтора» |

The English wording is the natural spoken English for that clock time, not a
translation of the Russian form. That is the comparison we want: how each
language actually says it.

Validator: passes. Oracle: 36/36.

### F14 — the loop detector is now the dominant terminator

`loop%` is 50-71% across arms, well over the 30% line the harness audit
warns at. But it is not causing the failures:

| arm | episodes killed as looping | of those, already successful |
|---|---|---|
| en | 10 | 6 |
| ru | 7 | 1 |
| ru_sys_ru | 7 | 2 |
| ru_translate_safe | 10 | 3 |

So the model finishes the work and then will not call `done`, and the detector
stops it. That is a real model behaviour, identical machinery in every arm, and
it should stay. What it costs is tokens: en 3011 tok/ep when looping against
2061 when stopping cleanly, ru 4114 against 2219.

Two consequences to keep in mind:

1. `steps_used` is capped by the detector in both arms, so `step_factor` = 1.02
   is partly forced. It is a lower bound on any real step effect.
2. English loops *more often* (71% vs 50%), which inflates English tokens, which
   pushes the CPST ratio **down**. The 2.18 is conservative on this axis.

### Reading of session 4

The cost story and the failure story have separated. The cost ratio is
directionally right but statistically empty at n=14. The +60 minute effect is
narrow, mechanistic, reproducible across three arms, and unfixable by any
mitigation tried so far. The probe now exists to measure it properly.

---

## 2026-08-18 — session 5

### time_probe.jsonl — F13 measured on the dedicated probe

Qwen2.5-3B, `tasks_time.json`, 18 single-action tasks, 1 repeat,
`--max-new-tokens 384`. Success by category:

| category | en | ru | ru_sys_ru | ru_translate_safe |
|---|---|---|---|---|
| `literal` (control) | 5/5 | 5/5 | 5/5 | 4/5 |
| **`into_hour`** | **7/7** | **0/7** | **0/7** | **0/7** |
| `subtractive` (control) | 0/3 | 0/3 | 0/3 | 0/3 |
| `arithmetic` | 2/3 | 2/3 | 2/3 | 2/3 |

Fisher exact, two-tailed:

* `into_hour`, en 7/7 against ru 0/7 — **p = 0.00058**
* `into_hour`, en 7/7 against the three Russian arms pooled 0/21 — **p = 8.4e-07**
* `literal`, en 5/5 against ru 5/5 — p = 1.0, as required
* `subtractive`, en 0/3 against ru 0/3 — p = 1.0, as required

The effect sits in exactly one cell and both controls behave exactly as they
must. English wrote all seven `into_hour` times **exactly** right, not one
minute off. The error distribution across the 21 Russian answers:

| error | count |
|---|---|
| **+60 min** | 16 |
| **−60 min** | 2 |
| +80 min | 1 |
| other (−355, −610) | 2 |

Eighteen of twenty-one are wrong by exactly one hour.

**Robustness check:** `alarm_exists` scans every alarm in the final state, and
the model creates 3-4 alarms per episode. Zero of the 21 Russian `into_hour`
episodes ever produced the correct time in *any* alarm. So the effect is not
an artefact of reading only the first value — the correct answer never appears.

### What the controls bought

`subtractive` is the important one. «без двадцати девять» / "twenty to nine"
fails in **all four arms**, English included (en wrote 02:30, 06:15, 21:10 for
08:40, 05:45, 18:50). That kills the lazy reading "Russian time expressions are
just harder". They are not. **One specific construction is** — the one where
the named ordinal is the hour being counted into — and English has no trouble
with its own idiom for the same clock time.

`literal` at 19/20 exact proves the probe is sound and the model can set alarms.

### F15 — correction: «полтора» is not a language effect

In v3_main, `a07` («за полтора часа до») looked like part of F13: English got
17:30, every Russian arm got 18:30. On the probe, `h16` is the same
construction and **all four arms wrote 18:30**, English included.

So English getting `a07` right was one lucky run, not a pattern. «An hour and a
half» / «полтора часа» is a **language-neutral** arithmetic failure: the model
subtracts 30 minutes instead of 90 in both languages. The `arithmetic` category
is 2/3 in every arm, identically.

F13 must therefore be stated narrowly: it is about the **counting-into-the-hour
construction**, not about Russian time arithmetic in general. The v3_main table
in session 4 overstated it by including a07. Corrected here.

This is also the first evidence that temperature 0 is not run-to-run stable —
the same construction gave a different English answer in two runs. Repeats are
needed for any rate claim.

### Do NOT quote the CPST number from this probe

`analyze` prints CPST ratio 1.874, CI [1.04, 4.43], which excludes 1.0 for the
first time in the project. **It is circular and must not be used.** The task set
was deliberately built around a construction that fails in Russian, so the
success gap is a property of the design, not a measurement of anything. A cost
ratio computed on a probe chosen for its failure mode measures the choice.

The cost claim still rests on `v3_main.jsonl`, where the CI is [0.75, 7.96] and
covers 1.0.

### F16 — the loop detector makes cost numbers from this probe meaningless

`loop%` is 89-100%. On tasks whose gold trace is **one action**, every arm
averaged:

| arm | alarms created | steps used |
|---|---|---|
| en | 3.7 | 6.3 |
| ru | 3.6 | 5.9 |
| ru_sys_ru | 3.6 | 6.2 |
| ru_translate_safe | 4.1 | 6.7 |

Gold is one alarm and one step. The model sets the alarm, does not call `done`,
re-sets it, and runs until the detector stops it. Six times the gold step count
and four times the gold state change.

The accuracy result is untouched by this — the model never wrote the right time
in any of those repeats. But every token, step and CPST figure from this probe
is dominated by the detector, not by the task. This is worth fixing before any
further cost claim: the harness has no cheap way to say "you are done".

### Reading of session 5

The accuracy finding is now real, narrow and correctly bounded: the
counting-into-the-hour construction fails in Russian at 0/21 against 7/7 in
English, p = 8.4e-07, with two controls confirming it is the construction and
not the language. Three mitigations do not touch it.

The cost finding is still nowhere near quotable, and this probe cannot help it.

### Next, in order of value

1. **Second model.** If Llama-3.2-3B shows the same +60 signature, the finding
   is about how models read the construction, not about Qwen. That is the
   difference between a bug report and a result. Highest value, cheapest run.
2. **A large model on a free tier.** Does scale fix it? Either answer is clean
   and publishable.
3. **Repeats at temperature 0.7** for a stability estimate, now that F15 has
   shown single runs move.
4. Fix the `done` problem before making any further cost claim (F16).

---

## 2026-08-18 — session 6

`meta-llama/Llama-3.2-3B-Instruct` is `gated=manual` on the Hub and there is no
`HF_TOKEN` on this machine, so the second model is **Phi-3.5-mini-instruct**
(3.8B, ungated, different family, different tokenizer, weaker Russian). For the
question being asked — is this Qwen or is this the construction — a different
family is what matters, and `ЧТО_ДЕЛАТЬ.md` §2 already asks for two models of
differing Russian quality.

### time_phi.jsonl — F13 replicates on a second model family

Phi-3.5-mini, `tasks_time.json`, 18 tasks, 1 repeat, 384 output tokens.
Static fertility 1.493.

| category | en | ru |
|---|---|---|
| `literal` (control) | 5/5 | 5/5 |
| **`into_hour`** | **7/7** | **0/7** |
| `subtractive` (control) | 0/3 | 0/3 |
| `arithmetic` | 3/3 | 2/3 |

Identical to Qwen in every cell that matters. What Phi wrote:

| task | want | en | ru |
|---|---|---|---|
| h01 | 07:20 | 07:20 | 08:20 (**+60**) |
| h02 | 14:30 | 14:30 | 08:00 (−390) |
| h03 | 20:05 | 20:05 | 21:05 (**+60**) |
| h04 | 05:15 | 05:15 | 08:15 (+180) |
| h05 | 10:10 | 10:10 | 11:10 (**+60**) |
| h06 | 06:30 | 06:30 | 07:30 (**+60**) |
| h07 | 13:25 | 13:25 | 14:25 (**+60**) |

Five of seven exactly +60. Across the plain `ru` arm of both models, **11 of 14
answers are wrong by exactly one hour**.

### F13 promoted: this is a property of the construction

Pooling the two independent model families on the `into_hour` category:

* English **14/14** correct, Russian **0/14** — Fisher exact two-tailed
  **p = 5.0e-08**
* including the two extra Russian arms run on Qwen: 14/14 against 0/28,
  **p = 1.9e-11**

Both controls hold in both models: `literal` 5/5 in every arm of both,
`subtractive` 0/3 in every arm of both. So the claim is bounded exactly where it
should be — the counting-into-the-hour construction fails in Russian, and it is
not that Russian time expressions are generally harder.

Two different tokenizers, two different training corpora, same one-hour error.
This is no longer a Qwen bug.

### F17 — CPST rewards confident failure

The Phi run has a clean harness audit for the first time in the project:
`loop%` 5.6% (en) and 0.0% (ru), against 89-100% on Qwen. Phi calls `done`.
So the cost numbers here mean something, and they say something unexpected:

| arm | success | steps | tok/ep | CPST |
|---|---|---|---|---|
| en | 83% | 1.83 | 1294 | 1553 |
| ru | 39% | 1.00 | 558 | 1435 |

`step_factor` 0.546 x `density_factor` 0.791 x `success_factor` 2.143 = **0.925**.
Bootstrap 95% CI [0.437, 2.648].

**Cost per successful task is slightly lower in Russian.** The mechanism is
visible in the step count: the Russian arm answers in exactly one step on every
single task, writes the wrong time, calls `done` and stops. The English arm
spends 1.83 steps because on the genuinely hard items (h03, h08, h09, h17, h18)
it retries. Russian is not expensive. Russian is **confidently wrong**.

This is a real problem for CPST as a metric, and it belongs in the paper rather
than being hidden:

> CPST divides total tokens by successes. A model that fails fast and cheaply in
> one language, without ever noticing it failed, scores *better* on CPST than a
> model that works harder and succeeds. The metric was designed to capture what
> a deployment pays, and it does — but it silently rewards confident error. The
> user whose alarm is set an hour late did not get a discount.

So the study now needs a second axis next to cost: not "how much did the
successful task cost" but "how often did the agent damage the user's data while
believing it had succeeded". `failure_kind == wrong_value` with
`called_done == True` is exactly that quantity and it is already logged.

### Where the project stands after session 6

Two results are now solid enough to write up:

1. **The counting-into-the-hour failure.** p = 5.0e-08 across two model
   families, both controls clean, three mitigations (Russian system prompt,
   naive translation, placeholder-protected translation) all fail to touch it.
   Silent, deterministic, invisible to every cost metric.
2. **Naive translate-then-act destroys user content**, 1/12 against 8/12,
   repaired to 7/12 by placeholder protection but still not profitable at
   2-4 steps.

The original cost hypothesis remains unsupported and now looks actively wrong in
the direction the tokenizer literature predicts: on the clean model the ratio is
0.925. The honest paper is not "non-English agents cost more". It is
**"non-English agents fail differently, and the standard cost metric cannot see
it"**.

### Still open

1. Does the effect survive on a large model? Free tier, zero cost. Now the most
   valuable remaining run: if a 70B model fixes it, the finding becomes "small
   models deployed for cheapness carry a correctness tax"; if it does not, the
   finding is much bigger.
2. Repeats at temperature 0.7 for stability, on Phi (clean) rather than Qwen.
3. The `wrong_value + called_done` rate as a headline number alongside CPST.
4. Qwen's inability to call `done` (F16) — a model property worth one paragraph,
   not a harness fix.

---

## 2026-08-18 — session 7

Groq free tier. `llama-3.3-70b-versatile` from the original plan is gone from
the catalogue; available large models are `openai/gpt-oss-120b` and
`qwen/qwen3.6-27b`. Both were run on the time probe, 18 tasks, arms en/ru,
1 repeat, 1024 output tokens. The second one is the more useful control: it is
the same family as the local Qwen2.5-3B, nine times larger and a generation
newer, so it isolates scale from family.

### F18 — scale fixes every category in the probe except this one

| model | literal | into_hour | subtractive | arithmetic |
|---|---|---|---|---|
| Qwen2.5-3B | 5/5 · 5/5 | **7/7 · 0/7** | 0/3 · 0/3 | 2/3 · 2/3 |
| Phi-3.5-mini | 5/5 · 5/5 | **7/7 · 0/7** | 0/3 · 0/3 | 3/3 · 2/3 |
| qwen3.6-27b | 5/5 · 5/5 | **7/7 · 2/7** | 3/3 · 3/3 | 3/3 · 3/3 |
| gpt-oss-120b | 5/5 · 5/5 | **7/7 · 2/7** | 3/3 · 3/3 | 3/3 · 3/3 |

(each cell is en · ru)

The two large models are **identical in every cell**. Going from 3-4B to
27-120B:

* `subtractive` goes 0/3 → 3/3 **in both languages**. The construction that
  defeated the small models regardless of language is simply solved.
* `arithmetic` goes to 3/3 in both, including «полтора часа» (F15).
* `literal` stays perfect everywhere.
* **`into_hour` in Russian goes 0/7 → 2/7 and stops there**, against 7/7 in
  English at every size.

So this is not "small models are bad at time". A thirty-fold increase in
parameters clears every other row of the table and leaves this one standing.

Pooled over all four models on `into_hour`: English **28/28**, Russian **4/28**,
Fisher exact two-tailed **p = 9.4e-12**.

Error signature across the four plain `ru` arms, n = 28: **+60 min in 18 cases**,
correct in 4, and six scattered errors.

### F19 — what the large models actually learned is the frequent surface form

The two items both large models get right are `h02` («половина третьего дня»)
and `h06` («половина седьмого утра»). Independently, both models get exactly
the same two right and the same five wrong.

The five they fail are the compositional ones: «двадцать минут восьмого»,
«пять минут девятого», «четверть шестого», «десять минут одиннадцатого»,
«двадцать пять минут второго».

That is a mechanism, not just a rate. «половина X» is a fixed, very frequent
phrase and is evidently memorised. The productive pattern «N минут X-ого»,
which a Russian speaker generates freely for any N and any hour, is not
learned as a rule — so scale buys coverage of common phrases and not the
grammar underneath. This is the sentence the paper should be built around.

### API backend caveat

`OpenAICompatBackend.count_tokens` is the crude `len // 4` fallback, so the
`fertility` line printed for these two runs (1.0) is meaningless. Real token
counts still come from the API `usage` field, so cost figures are fine; only
the static fertility ratio is not. Fix before publishing any fertility number
from an API run.

### Figures

`analyze.py` draws CPST and the three-factor decomposition — the right picture
for a hypothesis that is now dead. The accuracy result had no figure at all, so
`src/figure_main.py` is new:

* `figures/main.png` — success by category, one panel per model, 3B to 120B.
  The controls sit in the same picture as the effect, because the argument is
  carried by them.
* `figures/error.png` — the distribution of (written − correct) in minutes for
  the Russian arms, pooled. One bar at +60 doing all the work.

### Where the project stands after session 7

The headline is no longer about cost at all:

> Four models spanning 3B to 120B read the Russian counting-into-the-hour
> construction one hour late, while reading the equivalent English idiom
> perfectly. English 28/28, Russian 4/28, p = 9.4e-12. Scale solves every other
> time construction in the probe, in both languages, and does not solve this
> one; the residual competence at scale is confined to the single most frequent
> surface form. The failure is silent — the tool call succeeds, no error is
> raised — and invisible to every token-based cost metric.

Remaining work is depth, not breadth: more instances of the construction, a
stability estimate, and the parallel question for dates.

---

## 2026-08-18 — session 8

### F20 — literature check: the gap is real, but there is a rival explanation

This should have been done three sessions ago. Nothing published covers the
Russian counting-into-the-hour construction in LLMs. The closest work is
temporal-expression extraction (XLTime, HeidelTime) which is pre-LLM and about
*finding* time expressions rather than resolving them, and general temporal
reasoning benchmarks (TimeBench, Test of Time, PRIMETIME, Time-R1) which are
English-first and do not treat the language of the expression as a variable.

**But there is a documented phenomenon that a reviewer will raise immediately.**
Khodja et al. (2025) report that LLMs assign systematically higher likelihood to
*transitional* times — errors of exactly ±1 unit — than to the correct answer,
and hypothesise that transitional dates are over-represented in training data.
Fatemi et al. (2025) see the same ±1 concentration in duration questions.

So the obvious challenge to F13 is: **"you have rediscovered the known ±1
temporal bias."**

The data already answers this, and it is the controls that do it:

1. English is **28/28** on the *same clock times*. A general ±1 bias is not
   language-selective.
2. Russian `literal` («на 07:20») is 5/5 in every model. No ±1 there either.
3. The error is directional and grammatical: +60 in 18 of 28, matching exactly
   the misreading of the ordinal as the clock hour, not a symmetric ±1 spread.

This belongs in Related Work as a named alternative that the design rules out.
Without the literature check the paper would have had a hole where its strongest
paragraph now goes.

### `data/tasks_time_v2.json` — the expanded probe, built to test F19

24 single-action tasks. The counting-into-the-hour category is now **split by
how fixed the phrase is**, which is the direct test of the session-7 mechanism
claim:

| category | n | Russian form | prediction |
|---|---|---|---|
| `half_past` | 5 | «половина X» | large models hold it |
| `quarter_past` | 4 | «четверть X» | intermediate |
| `n_minutes` | 8 | «N минут X-ого» | fails at every scale |
| `literal` | 4 | digital | ~100% everywhere |
| `subtractive` | 3 | «без N» | control, and rules out ±1 bias |

If the Russian bars fall monotonically from `half_past` to `n_minutes` while
English stays flat, the finding stops being "Russian breaks" and becomes
**"models learn frequent surface forms and not the rule that generates them"** —
which is a claim about how these models represent language, not a bug report
about one construction.

`g16` («пять минут первого дня» = 12:05) is deliberate: the first hour after
noon runs 12:00-13:00, so a system applying the rule gets it right and a system
matching memorised phrases has nothing to match.

Validator: passes. Oracle: 48/48.

`src/figure_main.py` now orders the family from most fixed to fully productive,
so the predicted gradient reads left to right on the figure.

### Next run

```
python -m src.run_experiment --backend local --model Qwen/Qwen2.5-3B-Instruct \
    --tasks data/tasks_time_v2.json --arms en ru --repeats 1 \
    --max-new-tokens 384 --out results/v2_qwen3b.jsonl
```

then the same for `microsoft/Phi-3.5-mini-instruct`, and via Groq for
`qwen/qwen3.6-27b` and `openai/gpt-oss-120b`. 48 episodes each, single-action,
so roughly 10-20 minutes per local model and a few minutes per hosted one.

---

## 2026-08-18 — session 9

### F21 — the frequency gradient predicted by F19 is there

All four models on `tasks_time_v2.json`, 24 tasks, arms en/ru, 1 repeat.

| model | 07:20 literal | «половина» | «четверть» | «N минут» | «без N» |
|---|---|---|---|---|---|
| Qwen2.5-3B | 4/4 · 4/4 | 4/5 · **0/5** | 4/4 · **0/4** | 6/8 · **0/8** | 0/3 · 0/3 |
| Phi-3.5-mini | 4/4 · 4/4 | 5/5 · **1/5** | 3/4 · **0/4** | 8/8 · **0/8** | 0/3 · 0/3 |
| qwen3.6-27b | 4/4 · 4/4 | 5/5 · **4/5** | 4/4 · **0/4** | 8/8 · **0/8** | 3/3 · 3/3 |
| gpt-oss-120b | 4/4 · 4/4 | 5/5 · **4/5** | 4/4 · **1/4** | 8/8 · **0/8** | 3/3 · 3/3 |

(en · ru)

Pooled over the four models:

| category | English | Russian | Fisher exact, two-tailed |
|---|---|---|---|
| «половина X» — fixed phrase | 19/20 (95%) | 9/20 (**45%**) | p = 0.00125 |
| «четверть X» — intermediate | 15/16 (94%) | 1/16 (**6%**) | p = 8.6e-07 |
| «N минут X-ого» — productive | 30/32 (94%) | 0/32 (**0%**) | p = 6.1e-16 |
| whole family | 64/68 | 10/68 | p = 1.4e-22 |

**English is flat at 94-95% across all three. Russian falls 45% → 6% → 0%.**
The prediction from F19 was exactly this ordering, and it holds monotonically
in both large models: `half_past` 4/5, `quarter_past` 0/4 and 1/4, `n_minutes`
0/8 and 0/8.

**«N минут X-ого» is 0 out of 32 across four models spanning 3B to 120B**, on
clock times where English scores 30/32.

Error signature over the whole family, n = 68 per language:

* Russian: **+60 min in 33 cases**, correct in 10, then a scattered tail.
* English: **correct in 64 cases**, one +15, one +2, one +60.

That contrast is the answer to the Khodja ±1 objection in a single line: a
general ±1 temporal bias does not produce 64/68 exact in one language and 33
one-hour errors in the other on identical clock times.

### What this changes about the claim

The finding is no longer "Russian time expressions break". It is:

> Models learn frequent fixed time phrases and do not learn the rule that
> generates them. In Russian the productive construction «N минут X-ого» is
> read one hour late at every scale we tested, while the single most frequent
> member of the same family, «половина X», is recovered once the model is large
> enough. English, whose idiom for the same clock times names the completed
> hour, is unaffected throughout.

Scale moves the boundary between memorised and unlearned. It does not teach the
rule.

### Honest corrections to earlier sessions

1. **English is not perfect on the expanded set.** At 3B it is 4/5 on
   `half_past` and 6/8 on `n_minutes`. The perfect 28/28 of session 6 was a
   property of the smaller probe, not a law. English errors are mostly tiny
   (+15, +2 minutes) and are not the one-hour kind.
2. **`g16` is hard in both languages.** «пять минут первого дня» / "five past
   twelve in the afternoon" = 12:05. Qwen2.5-3B wrote 13:05 in *English* too.
   The hour that follows noon is a genuine edge case on both sides and should be
   reported as such rather than counted as evidence.
3. `subtractive` replicates session 6 exactly: 0/3 in both languages at 3-4B,
   3/3 in both at 27-120B.

### Figures

`figures/main.png` now orders the family from fixed to productive, so the
gradient reads left to right. `figures/error.png` pools the family, n = 68.

---

## 2026-08-19 — session 9: the expanded probe, and a citation I had to retract

### F21 — the frequency gradient, predicted then confirmed

`data/tasks_time_v2.json`, 24 tasks, four models, greedy decoding.

| model | literal | «половина» | «четверть» | «N минут» | «без N» |
|---|---|---|---|---|---|
| Qwen2.5-3B | 4/4 · 4/4 | 4/5 · 0/5 | 4/4 · 0/4 | 6/8 · 0/8 | 0/3 · 0/3 |
| Phi-3.5-mini | 4/4 · 4/4 | 5/5 · 1/5 | 3/4 · 0/4 | 8/8 · 0/8 | 0/3 · 0/3 |
| qwen3.6-27b | 4/4 · 4/4 | 5/5 · 4/5 | 4/4 · 0/4 | 8/8 · 0/8 | 3/3 · 3/3 |
| gpt-oss-120b | 4/4 · 4/4 | 5/5 · 4/5 | 4/4 · 1/4 | 8/8 · 0/8 | 3/3 · 3/3 |

Pooled: «половина» en 19/20 vs ru 9/20 (p = 0.00125); «четверть» 15/16 vs 1/16
(p = 8.6e-07); «N минут» 30/32 vs **0/32** (p = 6.1e-16). Whole family 64/68 vs
10/68, p = 1.4e-22.

The prediction was written down in session 7, before the probe existed, from
watching two large models solve exactly «половина» and fail every «N минут». It
came out monotone, independently, in both large models. That ordering is the
paper.

### F22 — frequency measured, and it refutes the easy reading

`src/frequency.py`, Google Books Ngrams, 1980-2019, mean over the bare phrase.

| category | freq RU | freq EN | acc RU | acc EN |
|---|---|---|---|---|
| «половина X» | 4.56e-08 | 3.97e-08 | 45% | 95% |
| «четверть X» | 2.60e-08 | 1.03e-08 | 6% | 94% |
| «N минут X-ого» | 8.28e-09 | 1.98e-09 | 0% | 94% |

I expected this to confirm "rare phrases are hard". It did the opposite, and the
result is better for it.

**Both languages fall in the same order, and English falls further** — 20x from
commonest to rarest against 5.5x in Russian. English accuracy does not move.
"Twenty five past nine" has **zero** occurrences in forty years of the corpus and
the large models still resolve it.

So rarity is not the explanation on its own. It bites in Russian only. The
reading: English exposes a rule that applies to unseen phrases, because the
numeral simply is the hour. Russian requires knowing that the surface number is
*not* the hour, and the models appear to hold only memorised instances. Where the
instance is common enough, they succeed; where it is not, there is no rule to
fall back on.

This is a dissociation, not a correlation, and it is much harder to argue with.

### F23 — stability under sampling

gpt-oss-120b, whole probe, three repeats at temperature 0.7.

- English: 100%, 100%, 100%. sd = 0.0 points.
- Russian: 50%, 46%, 46%. sd = 2.4 points.
- Gradient survives: `half_past` 60%, `quarter_past` 25%, `n_minutes` 4%
  (1 success in 24 sampled attempts).

Between-repeat variation is an order of magnitude smaller than the gap between
languages. The greedy numbers are not a knife edge.

### F24 — I cited two papers that I cannot show exist. Retracted.

Session 8 recorded Khodja et al. (2025) and Fatemi et al. (2025) on a systematic
±1 temporal bias, taken from a search summary. Trying to verify them, neither can
be located. They were almost certainly confabulated by the search summariser, and
I passed them on without checking.

**Both removed from the paper.** The ±1 objection is still worth answering, so it
is now stated as an anticipated objection and answered from our own data: English
is exact in 64 of 68 items on the same clock times, and Russian digital times are
read correctly by every model. No citation needed, and nothing is lost.

Rule going forward, added to the top of the paper's notes: **a reference gets
cited only after the actual paper has been opened.** A summary of a search result
is not a source.

### F25 — the real related work, verified by reading it

Mazzia, Pollastrini, Bernardi, Rubagotti, Amberti. *Benchmarking Multilingual
Temporal Reasoning in LLMs: The Temporal Reasoning Dataset.* IWSDS 2026,
pages 168-181. Amazon Alexa AI.

Read the PDF. Two facts matter:

1. Ten languages — en, de, es, fr, it, pt, nl, hi, ar, ja. **No Russian.**
2. Every time value in their carrier phrases is already digital: *"It is now
   19:29, what will the time be in 1 hour and 26 minutes?"* They test arithmetic
   over resolved clock values, never the reading of a spoken form.

German and Dutch are in TRD and both count into the coming hour. So the closest,
most recent, industrial multilingual temporal benchmark **contains the relevant
languages and would still miss this failure**, because of what it asks. That is
the cleanest possible statement of the gap, and it is verifiable in one page of
their paper.

Their own split between reasoning tasks (which transfer across languages) and
memorisation tasks (which do not) sits right next to F22 from the other side.

### Deliverables now in the repo

- `paper/paper.md` — full draft, every number in place, one verified citation and
  two clearly marked as unread.
- `figures/main.png`, `figures/error.png`, `figures/frequency.png`
- `src/frequency.py` — reproducible corpus measurement
- `outreach/emails.md` — rewritten for the actual audience

Remaining before sending: verify or cut the two marked references, publish the
repo, put the URL in §8.

### F26 — citation audit closed

All three references in `paper/paper.md` are now verified against the source, not
against a search summary.

| reference | how verified | what it supports |
|---|---|---|
| Mazzia et al., IWSDS 2026 | PDF read in full | ten languages without Russian; all carrier phrases give time in digits |
| Petrov et al., NeurIPS 2023, arXiv:2305.15425 | abstract read verbatim | up to 15x tokenization length differences on translated static text |
| Fatemi et al., arXiv:2406.09170, ICLR 2025 | abstract read | synthetic temporal-logic benchmark; structure, size, question type, fact order |

Note on the last one: **Bahare Fatemi is the first author of Test of Time.** That
is almost certainly where the phantom "Fatemi et al. 2025, ±1 temporal bias"
citation came from — a real author's name welded onto a claim that is not in the
paper. No claim about that paper's language coverage is made in the draft,
because I did not confirm it.

Standing rule from this: **a search result is not a source.** Open the paper.

---

## 2026-08-20 — session 10: chain-of-thought, and two things it exposed

### F27 — chain-of-thought does not touch the effect

New arms `en_cot` / `ru_cot`: identical task, but the system prompt asks the
model to reason step by step and to state explicitly how it converted the time,
inside `<think>` tags. The reasoning is stripped before parsing and logged.

The instruction is the same English sentence in both arms, so the only thing
that still differs is the language of the request.

Russian, fully productive `n_minutes`:

| model | base | with CoT |
|---|---|---|
| Qwen2.5-3B | 0/8 | 0/8 |
| Phi-3.5-mini | 0/8 | 0/8 |
| gpt-oss-120b | 0/8 | 0/8 |

**Zero out of twenty-four with chain-of-thought, across three models.** At 120B,
told explicitly to state the conversion, reasoning in Russian, the productive
form still never resolves.

Whole probe: Phi 5/24 → 5/24, gpt-oss 12/24 → 13/24, English unchanged at
24/24 for gpt-oss. Every movement is one item, which is noise.

This closes the first objection any reviewer raises. The model is not failing to
think. It is missing the rule.

### F28 — Qwen reasons in Chinese, and its CoT run is not comparable

Reasoning language by arm:

| model | `en_cot` | `ru_cot` |
|---|---|---|
| Qwen2.5-3B | **no traces at all** | **24 traces, all Mandarin** |
| Phi-3.5-mini | 15, English | 28, Russian |
| gpt-oss-120b | 17, English | 8 Russian, 3 English |

On English input Qwen ignores the instruction and answers directly; on Russian
input it complies but switches to Chinese. So its two arms are not doing the
same thing, and its +2 on English cannot be read as an effect of CoT.

**The `n_minutes` conclusion does not depend on it** — 0/8 either way — but the
Qwen row of any CoT table has to carry this caveat, or be re-run. Phi and
gpt-oss give the clean version and both reason in the language of the request.

Caught before it entered the paper. That would have been the sixth artifact.

Note: the Qwen CoT run also used the default step budget, while Phi and gpt-oss
ran with `--max-steps 4 --context-budget 3000` after the 3B model hit CUDA OOM
(context grows fast when every answer carries reasoning). Re-run Qwen with the
same flags before putting the three side by side.

### F29 — the models state the misreading out loud

The reasoning traces are the most direct evidence in the project. Phi, reasoning
in Russian, on `g02` («половина седьмого утра», 06:30):

> «Половина седьмого утра это **07:30** в 24-часовом формате.»

It sets 07:30. This is not a formatting slip. The model asserts the wrong
meaning as a fact, in the user's own language.

On `g06` («четверть шестого утра», 05:15):

> «Четверть шестого утра в 24-часовом формате это **08:15**.»

Qwen, reasoning in Chinese on `g10` («двадцать минут восьмого», 07:20):

> 用户请求在早上8点20分设置一个闹钟 — *the user is asking for an alarm at 8:20 AM*

And, remarkably, Qwen narrates its own translation chain on `g05`:

> 用户想在中文的"半点"设置一个闹钟，这需要转换成英文的"half past"
> *the user wants an alarm at the Chinese 半点, which needs converting to the
> English "half past"*

Russian in, Chinese reasoning, English idiom, wrong time out.

### F30 — five of my items were ambiguous, and the effect survives without them

The traces exposed a flaw in my own task file. Items phrased with «дня» as the
time-of-day marker are surface-ambiguous: «дня» is also the genitive of «день».
Phi read «двадцать минут третьего дня» as *the third day from today* and
computed a date, and «пять минут первого дня» as *the first day* → 00:05.

A native speaker does not make that reading. A model without the construction
does. Six of the seventeen into-hour items are affected: g01, g04, g09, g13,
g14, g16.

Robustness check across all four models:

| subset | English | Russian | Fisher exact |
|---|---|---|---|
| all 17 items | 64/68 | 10/68 | p = 1.4e-22 |
| 11 unambiguous items only («утра» / «вечера») | 42/44 | 8/44 | p = 2.9e-14 |

The effect does not move. This goes in the paper as a Robustness paragraph:
the objection is answered by the data rather than by argument.

It also raises the value of the human baseline, which is still not done. The
claim "a native speaker does not make that reading" is currently my intuition,
not a measurement.

### Cancelled: the second language

No access to a German or Ukrainian native speaker, and neither of us reads
either language. Building a probe in a language nobody involved can check would
inject exactly the asymmetry `validate_tasks.py` exists to prevent, and no test
would catch it. One language pair, stated plainly in Limitations.

The Discussion also has to be narrowed. «Halb acht» exists in German, Dutch,
Czech and the Scandinavian languages, but the fully productive «N минут X-ого»
appears to be Russian-specific — Polish, for instance, counts from the past hour
like English. The paper currently predicts the whole profile transfers. It
should predict only that the fixed forms do.

### F31 — the effect is not a property of one tool

`data/tasks_event.json`, generated by `tools/make_event_probe.py`: the same 24
time expressions asked as calendar events instead of alarms. The event title is
the same English identifier in both languages, so the only thing that differs is
still the time. Russian case had to be handled per category — «событие в
половин**е** третьего» is prepositional where the alarm frame takes accusative.

| model | alarm en | event en | alarm ru | event ru |
|---|---|---|---|---|
| Qwen2.5-3B | 18/24 | 17/24 | 4/24 | 4/24 |
| Phi-3.5-mini | 20/24 | 20/24 | 5/24 | 4/24 |
| gpt-oss-120b | 24/24 | 24/24 | 12/24 | 10/24 |

Counting-into-the-hour family, pooled over the three models:

| domain | English | Russian | Fisher exact |
|---|---|---|---|
| `set_alarm` | 47/51 | 6/51 | p = 2.5e-17 |
| `add_event` | 46/51 | 3/51 | p = 2.7e-19 |

English is flat across both domains. Russian is low in both. Both controls hold
in both.

**One correction to an earlier claim.** I had been writing that `n_minutes` in
Russian is "never solved". On the event domain gpt-oss-120b scores 2/8 on it.
The claim needs the domain attached, or both numbers reported. Zero is not a
constant; it is what that cell happens to be on the alarm probe.

Oracle on the event probe: 48/48. Validator: passes.

---

## 2026-08-21 — session 11: the p-values were never computed by anything

Nothing was run on the GPU. This session re-derived every number in the paper
straight from the logs, and one whole class of them did not survive.

### F32 — five p-values in the draft came from nowhere. Corrected.

Every count in the paper reproduces exactly: the gradient table cell by cell,
33 of 68 Russian answers wrong by +60, 64 of 68 English exact, the CoT zeros,
the event-domain table, the stability spreads. Those all come from the logs and
they are right.

The p-values do not come from the logs. **Nothing in this repository computed
them.** There was no significance test anywhere in `src/`. They were produced
outside the code and typed in.

| table | in the draft | actual | out by |
|---|---|---|---|
| «N минут» 30/32 vs 0/32 | 2.0e-13 | **6.1e-16** | 3 orders |
| whole family 64/68 vs 10/68 | 2.6e-13 | **1.4e-22** | 9 orders |
| unambiguous 42/44 vs 8/44 | 1.6e-12 | **2.9e-14** | 2 orders |
| `set_alarm` family 47/51 vs 6/51 | 2.0e-12 | **2.5e-17** | 5 orders |
| `add_event` family 46/51 vs 3/51 | 2.0e-12 | **2.7e-19** | 7 orders |

The tell was sitting in plain sight: the last two rows are different
contingency tables and were given **the same p-value to two significant
figures**. That cannot happen.

Everything from sessions 5 to 7 is correct — 0.00058, 8.4e-07, 5.0e-08,
1.9e-11, 9.4e-12 all reproduce to the digit, as do «половина» 0.00125 and
«четверть» 8.6e-07 from session 9. The bad values are exactly the five large
tables introduced in sessions 9 and 10. Whatever produced them stopped being
right somewhere past 1e-12 and started producing plausible-looking numbers.

**All five errors are conservative.** The true values are smaller, so the paper
understated its own result rather than overstating it. That is luck, not a
defence. A reviewer with R and thirty seconds would have found this, and the
finding would then have been "the author does not compute his own statistics".

This is the same failure as F24, in a different costume. There the fabricated
thing was a citation; here it is a number. Both times the pattern was identical:
something plausible arrived from outside the code and went into the draft
without a check. F24's rule was "open the paper". F32's rule is the same rule
for numbers.

### What changed so it cannot happen again

- `src/metrics.py` now has `fisher_exact()`. Exact rational arithmetic, so it
  does not lose precision at 1e-22 the way whatever produced the old numbers
  evidently did.
- Seven tests pin it to Fisher's own tea-tasting example (0.4857), to a worked
  table verified against R's `fisher.test` (0.002759), to symmetry under
  swapping the rows, and to the main result staying between 1e-23 and 1e-21.
  38 tests became 45.
- **`src/stats.py` is new and is now the only source of any statistic in the
  paper.** It regenerates the gradient table, the pooled comparisons, the error
  distribution, the robustness subset, the CoT table, the second domain, the
  stability spreads, the frequency table, and the first probe's historical
  numbers, all from the logs.
- `python -m src.stats --check paper/paper.md README.md NOTES.md HANDOFF.md`
  reads those documents and fails on any `p = ...` it did not itself compute.
  All four pass now. Run it before sending anything anywhere.

### F33 — three of the five task files were never validated

`tests/run_all.py` had a hardcoded list of two task files: `tasks_seed.json`
and `tasks_v2.json`. Both probes the paper actually rests on —
`tasks_time_v2.json` and `tasks_event.json` — plus `tasks_time.json` were added
later and were never checked by the cheap rule. For three sessions the guard
against a quietly-easier English half was not running on the main probe.

Validated by hand now: all five files pass, so nothing was actually wrong. The
runner globs `data/tasks_*.json` from here on, so a new file is covered the day
it is created rather than the day someone remembers.

### Two smaller corrections

- The draft said English has "three misses" out of 68 on the family. 68 − 64 is
  four. The fourth is −5 minutes. Fixed in §5.2.
- The stability spreads are sample standard deviations, n−1. `src/stats.py`
  says so explicitly, because computing them the other way gives 2.0 rather
  than 2.4 and the difference is not a mistake in either direction.

### Reading of session 11

Seven times now an apparent fact in this project has been the author's own
artifact: batching, the step ceiling, the list identifier, the step budget, the
output budget, the incomparable CoT arms, and now the significance tests. The
first six were caught by running something. This one was caught by recomputing
a number that had never been computed in the first place.

The result itself is unchanged and slightly stronger than it was this morning.

---

## 2026-08-21 — session 12: Polish, written down before it is run

### Why Polish is not a third language, it is a crossed design

The obvious next language was German, on the grounds that «halb acht» is 07:30.
That is a replication: another language that counts into the coming hour, giving
another instance of the same cell. Polish is worth more than that, because
Polish carries **both** constructions at once.

| clock | Russian | Polish | English | Polish counts |
|---|---|---|---|---|
| 07:30 | половина восьмого | **wpół do ósmej** | half past seven | **INTO the hour** |
| 07:15 | четверть восьмого | **kwadrans po siódmej** | quarter past seven | **FROM the hour** |
| 07:20 | двадцать минут восьмого | **dwadzieścia po siódmej** | twenty past seven | **FROM the hour** |

Russian counts into the coming hour in all three rows. Polish counts into the
hour in the first row and from the completed hour in the other two, exactly as
English does. So Polish holds the language, the speaker and the clock times
fixed and varies only the construction.

Two explanations of the Russian result are still standing, and they disagree
about what Polish does:

- **Ours.** Models memorise instances of the counting-into construction and
  never acquire the rule that generates them. Prediction: Polish «wpół do»
  behaves like Russian «половина», and Polish «po» behaves like English.
- **The rival.** Models are simply weaker on Slavic or on lower-resource
  languages. Prediction: both Polish cells fail.

Nothing else available at zero cost separates those.

### The prediction, recorded now, before any model has seen the file

Writing it down first is the only reason F19 → F21 is worth anything, so the
same discipline applies here.

1. Polish `digital` is near perfect. If it is not, our Polish is broken and
   nothing else in the file may be read.
2. Polish «wpół do» (n = 5 items, 20 observations over four models) lands near
   the Russian «половина» rate of 45 percent, and well below its own English
   half.
3. Polish «po» (n = 12 items, 48 observations) lands near the English rate of
   90 percent or better, against 4 percent for the Russian forms at the same
   clock times.
4. Within Polish, «po» separates from «wpół do».

**Power, computed before the run rather than after.** These are hypothetical
tables, not measurements, so they are written without the `p =` form on purpose:
`src/stats.py --check` should not accept them as reportable values. Against «po»
at 44/48, a Polish «wpół do» rate of 13/20 (65 percent) still separates at
about 0.011, and 9/20, the Russian rate, at about 7.9e-05. Anything at or above
14/20 does not separate, so an ambiguous outcome is possible and would be
reported as one.

Outcome 5 is the awkward one and is worth naming in advance: Polish could be
fine everywhere, «wpół do» included. That would not support the rival
explanation either, but it would mean the construction alone is not sufficient
and something about Russian specifically matters. It would be a real result and
it would need saying.

### What was built

- `tools/make_polish_probe.py` writes `data/tasks_time_pl.json`: the same 24
  clock times, a byte-identical English half, and a Polish half written out
  item by item rather than derived by substitution, so a Polish speaker can
  audit the probe by reading one table.
- New field `pl_direction` per task: `into_hour`, `from_hour`, `digital` or
  `subtractive`. This is the Polish construction, which is deliberately not the
  same grouping as `difficulty`, which stays the clock form so the Russian and
  Polish files remain comparable item for item.
- `src/stats.py` grows a Polish section, including the within-Polish
  comparison and a loud warning if Polish digital times come out below 90
  percent.

### Language plumbing: the arm no longer decides the language by guessing

`agent.py` computed `check_lang = "en" if arm in ("en", "en_cot") else "ru"`.
Under that line, Polish would have had to be stored in a slot named `ru`, and
every log line, every figure axis and every pooled count would have read "ru"
while holding Polish. Given F32 was a mislabelled number and F11 a mislabelled
parse status, that was not a corner to cut.

Now `ARM_LANGUAGE` maps each arm to a language code, `validate_tasks.py` reads
the second language out of the file instead of assuming Russian, `fertility()`
reports which language it measured, and the oracle replays the gold trace of
the language the arm actually reads. Adding a language is now a task file plus
two lines.

Regression: all five existing task files still validate, the Russian oracle is
still 144/144 on `tasks_time_v2.json`, and the 45 tests still pass.

Polish oracle: **144/144**. Validator: passes, `comparing en against pl`.

### Not yet done, and the paper must not pretend otherwise

The Polish wording was written from reference grammar by someone who does not
speak Polish. The forms are the textbook ones, and the design is partly
self-checking — a bad translation produces flat failure, not a clean split
between two constructions in the same language — but "textbook correct" is not
"what a Pole would say", and only the second supports the claim.

One wart, recorded rather than buried: «za dwadzieścia dziewiąta» is the normal
way to say 08:40, but «za dwadzieścia» alone can also mean "in twenty minutes",
and the nominative «dziewiąta» is what forces the clock reading. Russian «без
двадцати девять» has no such ambiguity. `subtractive` is a control in both
languages, so the main comparison is untouched, but a low Polish `subtractive`
score should be read with this in mind.

### The runs to do

```
python -m src.run_experiment --backend local --model Qwen/Qwen2.5-3B-Instruct \
    --tasks data/tasks_time_pl.json --arms en pl --repeats 1 \
    --max-new-tokens 384 --out results/pl_qwen3b.jsonl

python -m src.run_experiment --backend local --model microsoft/Phi-3.5-mini-instruct \
    --tasks data/tasks_time_pl.json --arms en pl --repeats 1 \
    --max-new-tokens 384 --out results/pl_phi.jsonl
```

then the same through Groq for `qwen/qwen3.6-27b` and `openai/gpt-oss-120b`,
and `python -m src.stats` reads all four.

### F34 — Polish, two small models: the wording is sound, the models are at floor

`data/tasks_time_pl.json`, Qwen2.5-3B and Phi-3.5-mini, greedy, 384 output
tokens. Cells are English · Polish.

| model | `digital` | «wpół do» INTO | «po» FROM | «za N» |
|---|---|---|---|---|
| Qwen2.5-3B | 4/4 · 4/4 | 4/5 · **0/5** | 10/12 · **4/12** | 0/3 · 0/3 |
| Phi-3.5-mini | 4/4 · 4/4 | 5/5 · **0/5** | 11/12 · **1/12** | 0/3 · 0/3 |

Pooled over the two: Polish `digital` **8/8**, «wpół do» **0/10** against English
9/10, «po» **5/24** against English 21/24, `subtractive` 0/6 in both languages.

**Prediction 1 passed and it is the one that mattered most.** Polish digital
times are 8/8. The carrier sentence works, the frame parses, the model can set
a Polish alarm. Whatever is failing above is about the time expression and not
about my Polish.

**Predictions 3 and 4 failed, and not in the rival's favour either.** Polish
«po» should have behaved like English, because it counts from the completed
hour exactly as English does. It scored 5/24 against English 21/24 on the same
clock times. The within-Polish comparison, «po» 5/24 against «wpół do» 0/10,
gives **p = 0.29**. Nothing is separated.

**This run does not distinguish the two explanations. It is a floor effect.**
Both models are simply unable to resolve Polish spoken clock forms of any kind.
That is not a surprise in hindsight and the project has the precedent written
down three sessions earlier: `subtractive` was 0/3 in **both** languages at
3-4B and 3/3 in **both** at 27-120B. A 3-4B model at floor on an unfamiliar
spoken-time form tells you about the model, not about the construction. F10 and
F11a are the same lesson and they cost a week.

So the Polish question is entirely on the two hosted models, and this run's job
was to prove the probe is sound before spending them. It did that.

### F34a — the error content, which is suggestive and is not a result

Accuracy says nothing here, but what the models actually wrote does split by
construction. Every Polish «wpół do» answer:

| task | want | phrase | model | wrote |
|---|---|---|---|---|
| g01 | 14:30 | wpół do trzeciej po południu | Qwen | 13:30 |
| g02 | 06:30 | wpół do siódmej rano | Qwen | 08:30 |
| g03 | 21:30 | wpół do dziesiątej wieczorem | Qwen | **22:30** |
| g04 | 11:30 | wpół do dwunastej przed południem | Qwen | 14:30 |
| g05 | 04:30 | wpół do piątej rano | Qwen | **05:30** |
| g01 | 14:30 | wpół do trzeciej po południu | Phi | **15:00** |
| g02 | 06:30 | wpół do siódmej rano | Phi | **07:00** |
| g03 | 21:30 | wpół do dziesiątej wieczorem | Phi | 19:00 |
| g04 | 11:30 | wpół do dwunastej przed południem | Phi | 14:00 |
| g05 | 04:30 | wpół do piątej rano | Phi | **05:00** |

In the five bold rows the written hour **is the named ordinal**. Phi drops
«wpół do» altogether and answers on the hour: «wpół do piątej» becomes 05:00,
«wpół do siódmej» becomes 07:00. Qwen keeps the half but hangs it on the wrong
hour: «wpół do dziesiątej» becomes 22:30, «wpół do piątej» becomes 05:30. Both
are the Russian mechanism — treat the named ordinal as the clock hour — in two
different surface shapes.

The «po» cell fails differently. Its errors are +15, −14, +300, −60, −10, +236,
+80, −10, +105, −15, −195, −80, −5, −40, +40, +170, −175 and one episode that
wrote no alarm at all. Scattered, mostly negative, **no +60 and no ordinal
pattern**. That is what incomprehension looks like next to what a misapplied
rule looks like.

**Do not promote this.** Ten answers, two models, both at floor. It is written
down because it was observed before the hosted runs, not because it carries
anything. If the large models reproduce the split on accuracy, this paragraph
becomes supporting detail. If they do not, this paragraph is noise and gets
deleted, and saying so now is the only thing that keeps it honest later.

### A harness note, so nobody reads the multi-alarm lists wrongly

Qwen stopped as `looping` on every Polish episode and wrote three or four
alarms each time, of which the repeats were almost always 07:30 regardless of
the item. That is F16, not a finding about Polish. `alarm_exists` scans every
alarm in the final state, so 0/5 on «wpół do» means the correct value appeared
in none of them, which is the robustness check that matters.

Polish fertility against English: 1.496 on the Qwen tokenizer, 1.462 on Phi.
Close to Russian's 1.4, so nothing new there.

### Still needed

```
python -m src.run_experiment --backend api --model qwen/qwen3.6-27b \
    --tasks data/tasks_time_pl.json --arms en pl --repeats 1 \
    --max-new-tokens 1024 --out results/pl_qwen27b.jsonl

python -m src.run_experiment --backend api --model openai/gpt-oss-120b \
    --tasks data/tasks_time_pl.json --arms en pl --repeats 1 \
    --max-new-tokens 1024 --out results/pl_gptoss120b.jsonl
```

These two decide it. On the main probe both scored `subtractive` 3/3 in both
languages and English 24/24, so if they are also at floor on Polish «po» the
rival explanation is live and the paper says so.

### F35 — the paper cut to the four-page target

Rewritten `paper/paper.md`. Every number in it now comes from `python -m
src.stats`, and `--check` passes on the file.

| | before | after |
|---|---|---|
| words counted against the limit | 2980 | **2502** |
| tables | 5 | **3** |
| full-width `figure*` floats | 3 | **1** |
| compiled ACL pages | 7 | to be confirmed |

What was cut, and why each was the right thing to lose rather than the easy
thing:

- **Related work, 597 to ~380.** Four papers at a paragraph and a half each
  became four tight paragraphs. Nothing was dropped, only the restatement.
- **§3 "The construction" folded into §1**, and its three-row example table
  became one sentence. The gloss is the argument; the table was decoration.
- **The frequency table deleted, the figure kept.** They carried the same three
  numbers twice. My own earlier note already said the figure carries §5.3
  better.
- **The probe-category table inlined as prose.** Five short rows costing a
  float.
- **Mitigations and the cost decomposition moved to Appendix A.** They use a
  different task set from the rest of the paper, and they carry the dead
  hypothesis. Appendices are outside the ACL page limit, so this is the
  cheapest half-page in the file and it loses no content.
- **Two of the three figures are now single-column.** Three `figure*` floats at
  `\textwidth` were probably the single largest reason the draft compiled to
  seven pages, and only Figure 1 actually needs the width.

What was added: the second domain, chain-of-thought, and the ambiguous-item
robustness check, all folded into one table in §4.4 rather than three sections
of prose. The §4.4 heading — "Four things that do not explain it away" — does
more work than the three separate subsections it replaces.

Two LaTeX bugs fixed in `build_tex.py` while regenerating. Limitations was
emitted as a numbered `\section`, which both misnumbers it and misrepresents
the ACL rule that it sits outside the page limit; it is now `\section*`.
Appendices were emitted as ordinary sections continuing the count, so Appendix
A would have printed as section 8; the generator now emits `\appendix` first.

**The page count is not confirmed.** There is no LaTeX on this machine, and
`build_pdf.py` uses its own layout, so its page count is not the ACL one. The
estimate from the reductions above is a little over four pages, not comfortably
under. Compile on Overleaf with XeLaTeX before believing any number. If it is
still over, cut in this order: the `subtractive` control row from Table 1, then
the last paragraph of §5, then Figure 3 — its one bar at +60 is already stated
in a sentence in §4.2.

Also needed for the Overleaf build and not in the repo: `acl.sty` and
`acl_natbib.bst` from acl-org/acl-style-files.

---

## 2026-08-21 — session 13: Polish on the hosted models, and the design working sideways

Both hosted runs done on the free Groq tier. gpt-oss-120b: 48/48 episodes, zero
parse errors, zero truncation, zero tool errors, zero looping, every episode
`done` in one step — the cleanest run in the project. qwen3.6-27b: same, with 5
truncated turns out of 48 episodes and no parse failures.

### F36 — the two large models are identical on Polish, and Polish is nearly solved

| model | `digital` | «wpół do» INTO | «po» FROM | «za N» |
|---|---|---|---|---|
| Qwen2.5-3B | 4/4 · 4/4 | 4/5 · **0/5** | 10/12 · **4/12** | 0/3 · 0/3 |
| Phi-3.5-mini | 4/4 · 4/4 | 5/5 · **0/5** | 11/12 · **1/12** | 0/3 · 0/3 |
| qwen3.6-27b | 4/4 · 4/4 | 5/5 · **5/5** | 12/12 · **12/12** | 3/3 · **1/3** |
| gpt-oss-120b | 4/4 · 4/4 | 5/5 · **5/5** | 12/12 · **12/12** | 3/3 · **1/3** |

(English · Polish.) The two large models agree in every single cell, exactly as
they did on the Russian probe in session 7.

On gpt-oss-120b the whole probe reads English 24/24, **Polish 22/24**, Russian
12/24. Both Polish misses are `subtractive`, and both are the ambiguity I wrote
down before the run: «za dziesięć siódma» (18:50) came back as **19:10**, so
«za» was taken as "in ten minutes" rather than "ten to". Russian «без десяти
семь» cannot be read that way. The control is compromised in Polish only, as
predicted, and it is the only place Polish fails.

**Prediction 2 was wrong.** I predicted Polish «wpół do» would land near the
Russian «половина» rate of 45 percent. It is 5/5 on both large models, exactly
right every time: 14:30, 06:30, 21:30, 11:30, 04:30. This was pre-registered as
"outcome 5, the awkward one", which is the only reason I am reading it as a
result instead of re-explaining it.

### F37 — the comparison that does work, and it is stronger than the one I designed

I framed Polish as a **within-Polish** contrast: «wpół do» against «po». At
scale both are at ceiling, so that comparison is empty (p = 0.59 pooled).

The informative contrast is **between Polish and Russian on the same clock
times**, because the two languages do not always use the same construction:

| clock form | Polish | Russian | relation |
|---|---|---|---|
| :30 | wpół do ósmej — counts INTO | половина восьмого — counts INTO | **matched** |
| :15 | kwadrans po siódmej — counts FROM | четверть восьмого — counts INTO | **crossed** |
| :20 | dwadzieścia po siódmej — FROM | двадцать минут восьмого — INTO | **crossed** |

Large models only, 27B and 120B:

| | Polish | Russian | Fisher |
|---|---|---|---|
| matched form, both count INTO | 10/10 | 8/10 | p = 0.47 |
| **crossed forms, same clock times** | **24/24** | **1/24** | **p = 1.6e-12** |

Over all four models the crossed comparison is 29/48 against 1/48, p = 1.6e-10.

**Where the two Slavic languages share the construction they tie. Where the
construction differs on identical clock times, they separate completely.**

No language-level or resource-level account survives that shape. "Models are
worse at Slavic" and "Polish is lower-resource than Russian" both predict one
language below the other everywhere. Instead the lower-resource language is the
one that succeeds, and only on the forms where its grammar counts from the
completed hour. The variable is the construction.

### What I got wrong, recorded plainly

Session 12 claimed Polish would separate the construction reading from the
"bad at Slavic" reading, and that "nothing else available at zero cost
separates those". Half right. It killed the resource reading decisively, and
by a better route than I planned. It cannot test the other half — the claim
that the *productive* counting-into form is never learned — because Polish has
no productive counting-into form. Its only counting-into phrase is the fixed,
frequent «wpół do», which is exactly the class Russian also solves at scale.

The design was right; my account of what it would show was too broad. The
pre-registration is what caught that, and it is the second time in three
sessions that writing the prediction down first has been the thing that kept a
result honest.

### One claim in the draft is now measured rather than asserted

§5 said the fully productive form "looks Russian-specific: Polish, for
instance, counts from the completed hour like English". That was a grammar
claim with no data behind it. It is now Polish «po» at 24/24 on the large
models, on the same clock times where Russian scores 0/16.

### Native-speaker check still outstanding

Everything above rests on Polish wording written from reference grammar by
someone who does not speak Polish. The `digital` control at 16/16 shows the
carrier sentence works, and the large models resolving 22 of 24 items shows the
phrases are intelligible, which is much stronger evidence than we had. But
"intelligible to a model" is not "what a Pole would say", and the paper should
not claim more than that until someone reads the 24 sentences.

---

## 2026-10-15 — session 14: review from Kyle Gorman and Richard Sproat

The August draft went to two people who have built exactly the systems this
paper is about. Both replied with substance. This entry records what they said
and what changed because of it.

### F38 — the p-values were reported in a form the test does not license

Kyle Gorman (CUNY Graduate Center), 13 October:

> "If you are using Fisher for null hypothesis testing, you shouldn't report
> miniscule p-values. Rather you should set an alpha-level (.05, or .01 are
> normal) and then report p < .01 [...] it doesn't matter HOW MUCH SMALLER THAN
> THE ALPHA-LEVEL a p-value is, it just matters WHETHER it's smaller."

He is right, and this is the second time the statistics in this project have
been wrong in a way the author could not see. F32 was arithmetic — five values
that no code computed. This one is conceptual: the arithmetic was correct and
the reporting was not. A frequentist test answers one question, whether the
result clears the threshold, and `p = 1.4e-22` on 68 items answers a question
nobody asked while inviting a reviewer to check a number that carries no
information.

**Changed.** `src/stats.py` gains `ALPHA = 0.01` and `format_p()`. Above the
threshold it prints two significant digits, below it `p < .01`. Two significant
digits rather than two decimal places, because at two decimals a p of .0114
prints as ".01" and reads as if the test landed exactly on the alpha level.

The exact value is still computed and still printed by the script in brackets,
because the notebook wants the record and `--check` needs it. It just does not
go in the paper. `NOTES.md` keeps exact values throughout; the public documents
do not.

**`--check` now enforces the convention as well as the arithmetic.** It flags
any `p = <value below alpha>` in a document and tells the author what to write
instead, and it verifies that a `p < X` claim is actually supported by a
computed value. Ten violations in `paper.md`, seven in `README.md`, five in
`HANDOFF.md`, eight in `tutorial.md`. All fixed.

Gorman also pointed at linguistic glossing conventions
(wellformedness.com, "How to write linguistic examples") and noted that Cyrillic
does not need the «» quotes since the script already sets it off, while the
Polish examples should be italicised. **Not yet done.**

### F39 — Sproat's data-volume numbers do not check out, and the claim they supported is gone

Richard Sproat, 14 October, on the obvious confound:

> "one assumes that these models have probably seen a great deal less Russian
> data in total than they have English data [...] I like to use the number of
> Wikipedia articles as a proxy. As of now, there are over 2.5 million English
> pages, 800K German articles and 320 Russian pages (even less than Dutch, which
> sits at 480K). Polish has 540K."

Checked against meta.wikimedia.org today:

| language | Sproat's figure | actual |
|---|---|---|
| English | 2.5M | 7.25M |
| German | 800K | 3.16M |
| Dutch | 480K | 2.23M |
| **Russian** | **320K** | **2.12M** |
| **Polish** | **540K** | **1.71M** |

Every figure is roughly a third of the current one, which looks like an old
snapshot quoted from memory. That part does not matter. **What matters is that
the ordering of the key pair is reversed**: he has Russian below Polish, and in
fact Russian has more.

This is load-bearing, because §4.4 said "Polish is the lower-resourced of the
two languages and it is the one that succeeds". On the one proxy available that
sentence is false, and it was repeated back to Sproat in correspondence.

**Changed.** The sentence is cut. What replaces it is better anyway, because it
does not depend on measuring corpus share at all:

> Whatever the two languages' relative share of the training corpus is, it is
> the same quantity in both halves of this comparison. A constant cannot produce
> an effect that appears on one clock form and vanishes on another.

The matched/crossed design *is* the control for data volume. Saying so directly
is stronger than any proxy, and it cannot be falsified by someone with better
numbers.

Wikipedia article count is a weak proxy for LLM training data in any case —
what matters is web-scale corpus composition, which is not public for any model
here. Limitations now names the confound and says plainly that we do not
measure it.

### F40 — every probe item is in the citation form

Sproat:

> "Двадцать минут восьмого will not always show up in that form, if for example
> the time expression is found in a context that requires a different case form
> (e.g. двадцати минутах восьмого). This adds further noise to the task."

Correct, and it is a genuine gap. The alarm frame takes one case throughout, so
the probe tests one surface form of a paradigm a speaker produces in several. A
model that resolves the citation form and fails the oblique ones would look
identical to ours. Added to Limitations; measuring it needs a probe that varies
the carrier frame, which does not exist yet.

### Still open after this session

1. **Few-shot prompting.** Both reviewers asked. Promised to Sproat. Not run.
   Prediction, written down now: two or three worked examples repair it, because
   the failure looks like a missing rule and a demonstration supplies the rule.
   If few-shot does *not* fix it, that is a much stronger result than the paper
   currently claims.
2. **German or Dutch.** Sproat's actual question: both count into the coming
   hour and both have more data than Russian — does the volume help? This is the
   experiment that separates "construction" from "construction plus scarcity",
   and Polish cannot answer it because it has no productive counting-into form.
3. Gorman's glossing conventions.
4. Native speaker for the Polish items. Still outstanding since August.

---

## 2026-10-15 — session 15: few-shot, written down before it is run

Both reviewers asked the same question, and it was promised to Sproat. The arms
exist, the oracle passes, nothing has been run on a model yet.

### The design, and the three ways it could have produced a fake result

**The demonstrated times are 09:05, 19:10 and 08:25, and none of them appears
in any probe.** Checked by a test, not by eye. A demonstration at a probe time
would let the model score by recalling what it was shown, and we would be
measuring recall of three items while calling it rule acquisition.

**All three languages demonstrate the same three clock times**, differing only
in wording. This is the dangerous one. `validate_tasks.py` guards the task
files, but the few-shot block lives in the prompt, where nothing looked. An
English block with easier or more numerous examples would reproduce the paper's
entire effect without any model doing anything unusual, and every existing
guard would have passed. `TestFewShotBlocks` now pins the parallelism: same
times, same order, same count, and no overlap with any probe.

**No reasoning inside the examples.** Input to output and nothing between.
Chain of thought is a separate arm, and mixing them would make it impossible to
say which one did the work.

49 tests. Oracle on the new arms: 48/48.

### Prediction, recorded now

Writing it down first is the only reason F19 → F21 and the Polish outcome mean
anything, so the same discipline applies.

**I expect three examples to repair it, at least on the large models.** The
failure looks like a missing rule rather than a missing capability, and a
demonstration supplies the rule directly. Specifically: Russian `n_minutes`
goes from 0/32 to somewhere above 20/32 on the two large models.

**If it does not repair it, that is the stronger result** and it changes the
paper's claim. A model that cannot apply the rule even when shown it three
times is not missing an instruction; something about the construction resists
the demonstration. That would be worth more than everything else in the paper.

**The awkward middle outcome to name in advance:** few-shot fixes the three
demonstrated minute-counts (five, ten, twenty-five past) and fails on the
others. That would be memorisation of the demonstrations rather than
acquisition of the rule, and it is detectable — the probe has eight
`n_minutes` items and the demonstrations cover only some of the minute values.
Split the result by minute-count before concluding anything.

### The runs

```
python -m src.run_experiment --backend local --model Qwen/Qwen2.5-3B-Instruct \
    --tasks data/tasks_time_v2.json --arms en_fewshot ru_fewshot --repeats 1 \
    --max-new-tokens 384 --out results/fs_qwen3b.jsonl

python -m src.run_experiment --backend local --model microsoft/Phi-3.5-mini-instruct \
    --tasks data/tasks_time_v2.json --arms en_fewshot ru_fewshot --repeats 1 \
    --max-new-tokens 384 --out results/fs_phi.jsonl
```

then the same through Groq for `qwen/qwen3.6-27b` and `openai/gpt-oss-120b`
with `--max-new-tokens 1024`, writing `results/fs_qwen27b.jsonl` and
`results/fs_gptoss120b.jsonl`. `python -m src.stats` reads all four.

### F41 — one of the four models no longer exists

`qwen/qwen3.6-27b`, a row in the main table since session 7, is gone from the
Groq catalogue as of 2026-10-15. The replacement at the same size is
`qwen/qwen3.8-27b`, a generation newer.

**That run can never be reproduced.** Everything we have of it is
`results/v2_qwen27b.jsonl`, `results/time_qwen27b.jsonl` and
`results/pl_qwen27b.jsonl` in this repository. This is the first time the
"keep every log, never a summary" rule has been the only thing standing
between the project and a lost result, and it is worth noting that hosted
models are not a stable substrate: a paper whose numbers live only in a
provider's current catalogue is not reproducible by anyone, including its
author.

Consequence for the few-shot run: comparing few-shot on qwen3.8 against
zero-shot on qwen3.6 would compare two different models and call it an effect
of the prompt. So qwen3.8-27b is being run on **both** conditions, keeping the
comparison within one model.

That substitution also buys something the paper did not have. The four models
in the main table are all from mid-2026 or earlier, and "does this persist in
the next generation?" is a question a reviewer will ask. qwen3.8-27b at the
same parameter count answers it directly.

### F42 — few-shot moves it, does not repair it, and the two large models disagree

Two hosted runs done. `qwen/qwen3.6-27b` is gone from the catalogue (F41), so
`qwen/qwen3.8-27b` was run on **all four arms in one pass**, keeping the
few-shot comparison inside one model.

**The manipulation was verified before the result was believed.** The logs do
not store the system prompt, so a null could have been a prompt that never
reached the model — exactly the shape of the six harness artifacts. First-turn
`prompt_tokens` rose by +122 (English) and +139 (Russian) against the base run.
The demonstrations arrived.

| model | ru «N минут» | ru whole family | en whole probe |
|---|---|---|---|
| gpt-oss-120b | 0/8 → **1/8** | 5/17 → 7/17 | 24/24 → 24/24 |
| qwen3.8-27b | 0/8 → **5/8** | 5/17 → 12/17 | 24/24 → 24/24 |

Pooled over the two: «N минут» **0/16 → 6/16**, p = .018. Whole family 10/34 →
19/34, p = .049. English sits at ceiling in both conditions on both models, so
the demonstrations did not simply teach the output format.

**My prediction was wrong.** Session 15 recorded: "I expect three examples to
repair it [...] somewhere above 20/32 on the two large models." That is above
60 percent; the actual is 6/16, under 40. Few-shot moves the productive form
off zero and does not fix it.

**The pre-registered memorisation check came out negative, which is the good
news.** Demonstrated minute counts 5/12, undemonstrated 1/4, p = 1.0. The
models are not simply replaying the three examples; qwen3.8 resolved the
undemonstrated "twenty past" once in two attempts. Whatever few-shot buys, it
is not recall of the demonstrations. The cell is thin — four undemonstrated
items — so this is a check that failed to fire, not a positive finding.

**The two large models disagree, and that is the part worth reporting.**
gpt-oss-120b barely moves, qwen3.8-27b goes to 5/8. A single number for "does
few-shot help" would hide this. The honest answer to Sproat's question is:
partially, and it depends on the model.

### F43 — the effect survives into the next generation

qwen3.8-27b is a generation newer than qwen3.6-27b at the same parameter
count, and it was not in the paper. Zero-shot, it reproduces the result:

- counting-into family: English **17/17**, Russian **5/17**, p < .01
- `literal` 4/4 both, `subtractive` 3/3 both — controls clean
- «половина» 5/5, «четверть» 0/4, «N минут» 0/8 — the gradient, again
  monotone, in a model that did not exist when the gradient was predicted

That last point matters. The ordering was written down in session 7 from
watching two models, confirmed in session 9 on four, and has now held on a
fifth released afterwards. Run audit: zero parse errors, zero truncation, zero
looping across all four arms.

### What this does to the paper

The claim gets more precise, not weaker. Before: models have not learned the
rule. Now: **handed the rule three times in worked examples, the strongest
model still resolves the productive form once in eight.** A missing
instruction would be repaired by an instruction. This is not.

Still needed before any of it is written up: the two local models, which
Aleksey is running.
