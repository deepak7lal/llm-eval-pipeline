# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A statistically-gated eval harness for Claude-powered features. Suites of
prompts live as YAML; a runner samples them against the Claude API, graders
score the outputs, and a gate decides whether CI goes red. It is a **CI/CD
pipeline**, not a library — the deliverable is a pass/fail signal people trust
enough to block a merge on.

The design premise everything follows from: **a pass rate is a sample, not a
measurement.** Run the same suite twice and the number moves. That is why cases
are sampled repeatedly, why scores carry confidence intervals, and why a
regression must be statistically significant before it fails a build.

## Layout

| Path | Role |
|---|---|
| `evals/config.py` | Single source of truth for model, effort, thresholds, repeats. Change defaults here, not inline. |
| `evals/client.py` | The only place that calls the Anthropic SDK for the model under test. Handles caching and retries. |
| `evals/cache.py` | Content-addressed on-disk response cache. Key = everything that can change an output. |
| `evals/cost.py` | Pricing table, USD accounting, run budget. |
| `evals/stats.py` | Wilson intervals, two-proportion test, power calculation. No scipy — all closed-form. |
| `evals/graders.py` | Scoring functions. Each returns `GradeResult(passed, score, detail)`. |
| `evals/runner.py` | Loads/validates YAML suites, samples cases concurrently, rolls trials up per case. |
| `evals/gate.py` | Floor + significance-tested regression + budget. Returns a `RunVerdict`. |
| `evals/compare.py` | A/B two prompt variants with a significance test. |
| `evals/report.py` | Markdown scorecard (PR comment) and `results.json` (artifact). |
| `evals/cli.py` | `python -m evals [run\|compare\|power\|cache]`. |
| `evals/datasets/*.yaml` | The suites. One file per suite. |
| `tests/` | Unit + end-to-end tests. **Must never call the API** — stub `evals.runner.complete`. |

## Vocabulary — keep these straight

- **Case** — one input plus its graders. Identified by `id`.
- **Trial** — one sample of one case. `repeats: 3` turns 6 cases into 18 trials.
- **Pass rate** — successes / **trials**, not cases. This is what the gate reads.
- **Flaky** — a case that passed on some trials and failed on others. Reported
  separately from a clean failure, because it means something different.

## Commands

```bash
pip install -e ".[dev]"                 # setup
pytest                                  # offline tests, no API key - run these first
python -m evals run                     # full run; exit 1 if the gate fails
python -m evals run --suite classification --repeats 5
python -m evals run --no-gate           # score without failing
python -m evals run --update-baseline   # only ever on main
python -m evals compare --variant terse # A/B a prompt variant
python -m evals power --effect 0.05     # trials needed to detect a 5-point drop
python -m evals cache --clear           # drop cached responses
```

## API conventions — do not drift from these

- **Model is `claude-opus-5`.** Never downgrade to Sonnet or Haiku for cost;
  scores are not comparable across models, and a model change invalidates the
  baseline. Changing the model means regenerating the baseline.
- **Adaptive thinking**: `thinking={"type": "adaptive"}`. `budget_tokens` is
  removed on this model and returns a 400.
- **Effort** goes in `output_config={"effort": ...}`, not top-level.
- **No assistant prefill** — it returns a 400. Use structured outputs or the
  system prompt to shape the response.
- **Structured outputs** use `client.messages.parse(..., output_format=Model)`
  with a Pydantic model (see `graders.llm_judge`), not hand-parsed JSON.
- **Prompt caching**: the suite's system prompt carries
  `cache_control: {"type": "ephemeral"}` and is sent first. Keep volatile
  content (the per-case input) after it — a byte change in the prefix
  invalidates the whole cache and quietly multiplies the bill.
- **Refusals are results, not errors.** `stop_reason == "refusal"` is recorded
  as a failed trial and is never cached. Do not add a fallback model to route
  around it — that silently changes what is being measured.

## The gate

Three independent failure modes, all in `evals/gate.py`:

1. **Floor** — pass rate below `min_pass_rate` (80%).
2. **Regression** — dropped more than `max_regression` (5 points) against the
   baseline **and** the drop is significant at `ALPHA` (0.05). Both conditions
   are required. A drop that fails only the first is reported under "movement
   within noise" and does not block.
3. **Budget** — the run cost more than `EVAL_BUDGET_USD` ($5). A prompt change
   that triples cost is a regression too.

Suites below `min_cases_to_gate` (5 trials) are reported but never gate.

**Do not "fix" a red gate by lowering a threshold or refreshing the baseline.**
If a regression looks like noise, the answer is more trials (`--repeats`), not a
wider allowance. `python -m evals power` says how many you need.

## Adding a suite

1. Drop a YAML file in `evals/datasets/`. Required: `name`, `system`, `cases`;
   each case needs `id`, `input`, `graders`. Optional: `effort`, `repeats`,
   `max_tokens`, `variants`.
2. Run `pytest tests/test_datasets.py` — it validates grader names, unique ids,
   and that every case is graded.
3. Run the suite alone, confirm the scores are sane, then let CI on `main`
   record the baseline. **Never hand-edit `baselines/main.json`.**

## Adding a grader

Add a function to `evals/graders.py` with the signature
`(output: str, spec: dict, case: dict) -> GradeResult`, register it in
`GRADERS`, and add unit tests. Prefer deterministic graders; reach for
`llm_judge` only when the output is genuinely free-form — it costs a second API
call per trial and adds its own variance on top of the model's.

## Prompt variants

Variants are alternative system prompts declared under `variants:` in a suite.
`baseline` is reserved for the suite's own `system`. Compare with
`python -m evals compare --variant <name>`; it runs both arms over the same
cases and reports whether the difference clears significance. Comparisons never
gate — they inform a decision, they do not make one.

Do not promote a variant on a single run with `repeats: 1`. The comparison
report prints the trial count needed to resolve a difference of the size you
observed; respect it.

## Cost

Every run that touches the API costs real money.

- The response cache (`.eval-cache/`) means unchanged prompts are free on
  re-runs. It is keyed on model, effort, system, input, max_tokens, and repeat
  index — any prompt edit misses by construction, so a hit is always valid.
- When iterating on harness code, use the stubbed end-to-end tests
  (`tests/test_end_to_end.py`) rather than live calls — monkeypatch
  `evals.runner.complete`.
- `results.json` records per-suite token counts and USD if you need to see what
  a run actually cost.

Keep `PRICING` in `evals/cost.py` current. It is the one table in this repo
where a stale number costs money rather than just being wrong.
