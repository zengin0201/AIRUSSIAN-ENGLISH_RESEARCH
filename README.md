# One Hour Late: LLMs Memorise Russian Clock Phrases Instead of Learning the Rule

**Four models, from 3B to 120B parameters, set a Russian user's alarm exactly one hour late. The tool call succeeds. No error is raised anywhere.**

Environment, probe, all run logs, figures and the full lab notebook (`NOTES.md`) are in this repository. The short version follows.

Russian says «двадцать минут восьмого» for 07:20. The ordinal names the hour being counted *into*, so this is twenty minutes **of the eighth hour**. English "twenty past seven" names the hour already completed. Every model we tested reads the English idiom correctly and the Russian one an hour late.

| | English | Russian |
|---|---|---|
| counting-into-the-hour expressions | **64 / 68** | **10 / 68** |

Fisher exact, two-tailed: **p < .01**. Of the 68 Russian answers, **33 are wrong by exactly +60 minutes**. English is exactly right in 64 of them, on the same clock times.

## Why this is not just "small models are bad at time"

The probe has two control categories, and they carry the argument.

| model | `literal` | «половина» | «четверть» | «N минут» | «без N» |
|---|---|---|---|---|---|
| Qwen2.5-3B | 4/4 · 4/4 | 4/5 · **0/5** | 4/4 · **0/4** | 6/8 · **0/8** | 0/3 · 0/3 |
| Phi-3.5-mini | 4/4 · 4/4 | 5/5 · **1/5** | 3/4 · **0/4** | 8/8 · **0/8** | 0/3 · 0/3 |
| qwen3.6-27b | 4/4 · 4/4 | 5/5 · **4/5** | 4/4 · **0/4** | 8/8 · **0/8** | 3/3 · 3/3 |
| gpt-oss-120b | 4/4 · 4/4 | 5/5 · **4/5** | 4/4 · **1/4** | 8/8 · **0/8** | 3/3 · 3/3 |

Each cell is English · Russian.

- **Digital times are perfect everywhere.** The probe works and the models can set alarms.
- **Subtractive forms** («без двадцати девять» / "twenty to nine") defeat the small models **in both languages**, and the large models solve them in both. So Russian time expressions are not simply harder.
- **Scale fixes everything except one column.** Going from 3-4B to 27-120B solves the subtractive forms in both languages. Russian «половина» climbs from 0-1/5 to 4/5. Russian «N минут» stays at **0/8 at every size**, while English holds 94% throughout.

## Within Russian, accuracy follows how common the phrase is

Splitting the construction by how fixed the phrase is gives a clean gradient, pooled over four models:

| category | English | Russian | Fisher exact |
|---|---|---|---|
| «половина X» — the common fixed phrase | 19/20 (95%) | 9/20 (45%) | p < .01 |
| «четверть X» — intermediate | 15/16 (94%) | 1/16 (6%) | p < .01 |
| «N минут X-ого» — fully productive | 30/32 (94%) | **0/32 (0%)** | p < .01 |

Both large models produce that ordering independently. English stays flat.

## And rarity is not the explanation

The obvious reading is that rare phrases are hard. We measured corpus frequency for every phrase (Google Books Ngrams, 1980–2019, `src/frequency.py`) to check it.

| category | frequency RU | frequency EN | accuracy RU | accuracy EN |
|---|---|---|---|---|
| «половина X» | 4.56e-08 | 3.97e-08 | 45% | 95% |
| «четверть X» | 2.60e-08 | 1.03e-08 | 6% | 94% |
| «N минут X-ого» | 8.28e-09 | 1.98e-09 | 0% | 94% |

Both languages get rarer in the same order, and **English falls further**: 20x from commonest to rarest, against 5.5x in Russian. English accuracy does not move. "Twenty five past nine" has zero occurrences in forty years of the corpus, and the large models still resolve it correctly.

So frequency predicts Russian accuracy and predicts nothing about English accuracy. That is a dissociation, not a correlation.

Our reading: English hands the model a rule that works on a phrase it has never seen, because the numeral in "twenty five past nine" simply is the hour. Russian hands it a construction where the surface number is *not* the hour, and the models behave as if they hold only the instances they have met. Where the instance is common, they succeed. Where it is not, there is no rule underneath to fall back on.

**Scale adds memorised instances. It does not add the rule.**

## Polish: the same construction, a different language

The English-versus-Russian contrast confounds two things. English is both the
higher-resourced language and the one that counts from the completed hour, so
"models are worse at Russian" cannot be told apart from "models are worse at
this construction".

Polish separates them, because Polish uses **both** constructions:

| clock | Polish | Russian | English |
|---|---|---|---|
| 07:30 | **wpół do ósmej** — counts INTO the hour | половина восьмого — INTO | half past seven — from |
| 07:15 | kwadrans po siódmej — **from** | четверть восьмого — INTO | quarter past five — from |
| 07:20 | dwadzieścia po siódmej — **from** | двадцать минут восьмого — INTO | twenty past seven — from |

So the half-past form is matched between the two Slavic languages and the other
two are crossed. On the two large models:

| | Polish | Russian | Fisher exact |
|---|---|---|---|
| matched form, both count INTO the hour | 10/10 | 8/10 | p = .47 |
| **crossed forms, identical clock times** | **24/24** | **1/24** | **p < .01** |

**Where the two Slavic languages share the construction they tie. Where the
construction differs they separate completely.** gpt-oss-120b reads the whole
Polish probe at 22/24 against 12/24 in Russian. The design controls for
training-data volume without measuring it: whatever the two languages'
relative share of the corpus is, it is the same quantity in both halves of
the comparison, and a constant cannot produce an effect that appears on one
clock form and vanishes on another.

Polish also corrected us. We had predicted that any language counting into the
coming hour would show the Russian profile. Polish «wpół do» counts into the
hour and is answered 5/5 — because it is a fixed, frequent phrase, like Russian
«половина». The failure needs *both* properties: the surface number must not be
the hour, **and** the phrase must be productive enough that the exact instance
was never memorised. Polish has no productive counting-into form, so it cannot
test that half of the claim.

**Caveat, stated up front.** The Polish wording was written from reference
grammars by an author who does not speak Polish. The digital control is 16/16
and the large models resolve 22 of 24 items, so the sentences work — but that is
not a native speaker calling them natural, and that check is still outstanding.
One control is known to be compromised: «za dwadzieścia dziewiąta» is the
ordinary way to say 08:40, but «za dwadzieścia» alone can also mean "in twenty
minutes", and both large models took the second reading. That is where both
Polish misses fall.

## Chain-of-thought does not help

Asked to reason step by step and to state explicitly how it converted the time,
the Russian productive form still scores **0/8 on each of three models, 0/24
pooled**. At 120B, reasoning in Russian, told to state the conversion, it never
resolves. The traces show the models asserting the wrong meaning outright — Phi,
reasoning in Russian on «половина седьмого утра» (06:30), writes that this is
07:30, and sets 07:30.

The models are not failing to think. They are missing the rule.

## It is not the tool either

The same 24 expressions asked as calendar events instead of alarms
(`data/tasks_event.json`) reproduce the effect: pooled over three models,
English 46/51 against Russian 3/51 on the counting-into family, p < .01,
against 47/51 versus 6/51 on the alarm probe.

## Three mitigations that do not work

Tested as separate experimental arms, all on the same tasks:

| mitigation | result |
|---|---|
| Russian system prompt | same +60 error |
| translate to English first | same +60 error, and the translation corrupts the time before the agent starts |
| translate with the user's values protected by placeholders | same +60 error |

Placeholder protection repairs *content* damage (see below) but not this. The meaning lives in the sentence, not in a quoted span.

## Second result: naive translate-then-act destroys user data

A common deployment shortcut is to translate the request into English and run the agent in English. On our seed tasks:

| arm | success | cost per successful task |
|---|---|---|
| Russian, as-is | 8/12 | 3,641 tokens |
| translate to English first | **1/12** | 31,321 tokens |
| translate, values protected | 7/12 | 5,428 tokens |

All 11 failures of the naive arm are `wrong_value`: the agent did the work and wrote the wrong words into the user's data. «спортзал» became `gym`, «билет» became `ticket`. Placeholder protection repairs almost all of it, but at 2-4 step tasks the policy still costs about 19% more per successful task for no benefit.

## What we set out to measure, and why we stopped

The project began as a cost study: how much more does one *successfully finished* task cost in Russian, and is the gap larger than the tokenizer ratio predicts? We measured cost per successful task (CPST) and split it into step, density and success factors.

**The cost hypothesis did not hold.** Every honest confidence interval covers 1.0:

| task set | CPST ratio RU/EN | 95% CI |
|---|---|---|
| seed (12 tasks) | 1.08 | [0.53, 2.10] |
| traps + long chains (14 tasks) | 2.18 | [0.75, 7.96] |
| time probe, clean model | 0.93 | [0.44, 2.65] |

Two reasons, both worth reporting:

1. **The tokenizer tax is diluted.** The English system prompt and tool schema are about 700 tokens and identical in every arm. The user's sentence is a small slice of that, so a fertility ratio of 1.4 becomes a density factor of about 1.03.
2. **CPST rewards confident failure.** On the cleanest model, the Russian arm answers in exactly one step, writes the wrong time, calls `done` and stops. The English arm spends 1.83 steps because it retries the hard items. Russian therefore scores *cheaper* per successful task. The user whose alarm is an hour late did not get a discount.

So the honest headline is not "non-English agents cost more". It is **"non-English agents fail differently, and the standard cost metric cannot see it"**.

## Design decisions worth defending

**No LLM judge.** Success is decided by comparing the final environment state against a fixed list of checks. LLM judges are known to score non-English answers differently, which would mix that bias into the effect we measure.

**The English wording is natural spoken English for that clock time, not a translation of the Russian form.** We compare how each language actually says it. Any other choice measures translation, not comprehension.

**Controls sit in the same figure as the effect.** The claim is "this construction, not this language", and only the controls can carry it.

**Single-action probe tasks.** Each probe task is one `set_alarm` call with one check on the time. An earlier version bundled the trap with a second unrelated action, and a failure there hid the thing we were trying to observe.

**The oracle test.** `--backend oracle` replays a known-correct action trace. If it is not 100%, the environment or the checks are broken, not the model.

**Task symmetry is validated statically.** `src/validate_tasks.py` checks that the English and Russian halves of a task have the same checks in the same order, the same times, dates and list names, and the same sequence of tool calls in the gold trace. Only the human-readable words may differ. This is the guard against the worst failure mode in this kind of study: an English task that is quietly easier than its Russian twin.

**A cut-off answer is not a parsing failure.** If a batch of tool calls runs past the output budget it is sliced mid-string and the turn yields zero actions, not a partial plan. We log that separately (`finish_reason == "length"`). An earlier run reported 32 "malformed JSON" turns; every one of them was a correct plan cut at the token limit.

**Cost is computed, not paid.** We log real token counts and multiply by public list prices.

## Install

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python -m tests.run_all
```

45 tests plus validation of every task file. Two seconds, no GPU. Run it before every experiment and after touching any task file.

```bash
python -m src.run_experiment --backend oracle --tasks data/tasks_time.json --arms en ru --out results/oracle.jsonl
```

```bash
python -m src.run_experiment --backend local --model Qwen/Qwen2.5-3B-Instruct --tasks data/tasks_time.json --arms en ru --repeats 1 --max-new-tokens 384 --out results/time_probe.jsonl
```

For a hosted model, set `ALT_API_BASE` and `ALT_API_KEY` and use `--backend api`.

```bash
python -m src.figure_main --probe results/time_probe.jsonl results/time_phi.jsonl --tasks data/tasks_time.json --out figures/
```

Every number quoted above is regenerated from the logs by one script, which
also checks this file for any p-value it did not itself compute:

```bash
python -m src.stats --check README.md
```

## Files

```
src/environment.py     tools, state, deterministic checks
src/frequency.py       corpus frequency of every probe phrase (Google Ngrams)
src/stats.py           every statistic in the paper, regenerated from the logs
src/agent.py           the loop, the prompts, the five arms
src/backends.py        local 4-bit model, OpenAI-compatible API, oracle
src/metrics.py         CPST, fertility, decomposition, bootstrap CI
src/run_experiment.py  CLI runner
src/figure_main.py     the paper's three figures
src/analyze.py         cost tables and the three-factor decomposition
src/analyze_sweep.py   the output-budget sweep
src/validate_tasks.py  checks that the EN and RU halves are the same task
src/energy.py          GPU power sampling
tests/                 45 stdlib tests, no pytest, no GPU
data/tasks_time_v2.json  24 single-action time expressions — the main probe
data/tasks_time.json   18-item first version of the probe
data/tasks_event.json    the same 24 expressions asked as calendar events
data/tasks_time_pl.json  Polish mirror: the crossed-construction control
data/tasks_v2.json     14 tasks: 8 translation traps, 6 long chains
data/tasks_seed.json   12 general parallel EN/RU tasks
NOTES.md               lab notebook: every run, every number, findings F1-F33
```

## Every number here is generated, not typed

```bash
python -m src.stats --check README.md
```

`src/stats.py` regenerates every statistic in this README and in the paper
straight from the logs, and `--check` fails on any p-value in a document that it
did not itself compute. It exists because an earlier draft carried five
p-values that no code in this repository had ever produced; they were wrong by
up to nine orders of magnitude. See `NOTES.md`, F32.

## Limitations

- **Frequency is measured on written text.** Google Books under-represents spoken clock phrases. A spoken corpus would be a better instrument.
- **The probe is small.** Seventeen counting-into-the-hour items per model. The separation is complete and the p-value reflects that, but the rate estimate is thin. Expanding to about 20 items, split by sub-construction, is the next planned run.
- **The stability check covers one model.** Rerunning the whole probe three times on gpt-oss-120b at temperature 0.7 gives English 100/100/100 (sd 0.0 points) and Russian 50/46/46 (sd 2.4 points), with the gradient intact. The other three models were run greedily and once.
- **The Polish wording has not been checked by a native speaker.** It was written from reference grammars. The controls say the sentences work; they do not say the sentences are natural.
- **Three languages, all Indo-European**, and only Russian supplies a productive counting-into construction. Fixed counting-into phrases also exist in German, Czech, Danish, Swedish and Dutch; on the Polish evidence we expect those to be memorised and solved rather than to fail, but we have not tested them.
- **The environment is simulated**, so results may not transfer to real APIs with real latency and rate limits.
- **Qwen2.5-3B often does not call `done`** and is stopped by our loop detector in 50-100% of episodes. This inflates its token counts. It does not affect the accuracy result — the model never wrote the correct time in any of those repeated attempts — but cost figures from that model should be read with care.
- **The API backend's `count_tokens` is a crude fallback**, so the static fertility ratio printed for hosted runs is meaningless. Real token counts come from the API usage field and are fine.

## License

MIT for the code. The task files are released under CC-BY-4.0.
