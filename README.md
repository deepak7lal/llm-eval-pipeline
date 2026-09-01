# llm-eval-pipeline

Statistically-gated evaluation harness for Claude-powered features. Suites of
prompts live as YAML, run against the Claude API on every pull request, and gate
the merge when quality actually drops — not when it merely looks like it did.

## Why this exists

Most LLM eval setups fail in one of two ways. They fire on every wobble until
the team learns to ignore them, or they never fire at all. Both come from
treating a pass rate as a measurement. It isn't — it's a sample, and samples
need error bars.

So this harness samples each case more than once, reports scores with confidence
intervals, and fails a build only when a regression is both **large enough to
matter** and **statistically real**.

## How it works

```
evals/datasets/*.yaml  ->  runner  ->  graders  ->  gate  ->  scorecard
   suites + cases          N samples   pass/fail   floor +    PR comment
   + variants              per case    + score     stats +    + artifact
                           (cached)                budget
```

A pull request fails when a suite drops below the 80% floor, regresses more than
5 points against `main` **with p < 0.05**, or the run blows the cost budget. The
scorecard is posted as a sticky PR comment either way.

## Sample scorecard

```markdown
## LLM eval scorecard - [FAIL]

`claude-opus-5` at effort `medium` - 22/30 trials passed across 1 suite(s).
Cost: $0.1056 of a $5.00 budget.

| Suite            | Pass rate | 95% CI     | Baseline | Delta  | Trials | Flaky | Status |
|------------------|----------:|:-----------|---------:|-------:|-------:|------:|:-------|
| `classification` |     73.3% | [56%, 86%] |    95.0% | -21.7% |  22/30 |     1 | [FAIL] |

### Why the build failed
- `classification`: pass rate 73.3% is below the floor of 80%
- `classification`: regressed 21.7% against baseline 95.0% (allowance 5%, p=0.038)

### Flaky cases
- `classification` / `billing-disguised-as-bug` - 2/5 passed
```

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env        # then set ANTHROPIC_API_KEY
```

Credentials resolve from `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or an
`ant auth login` profile — in that order. Nothing is hardcoded. In CI, add the
key as the repository secret `ANTHROPIC_API_KEY`.

## Usage

```bash
pytest                                   # offline tests, no API key needed
python -m evals run                      # run everything, exit 1 if the gate fails
python -m evals run --suite classification --repeats 5
python -m evals run --effort high        # override effort for this run
python -m evals run --no-gate            # report scores, always exit 0

python -m evals compare --variant terse  # A/B a prompt variant
python -m evals power --effect 0.05      # trials needed to detect a 5-point drop
python -m evals cache                    # cache stats; --clear to drop it
```

Reports land in `reports/`: `scorecard.md` (human) and `results.json` (machine —
per-case trials, intervals, p-values, tokens, USD).

## Writing a suite

```yaml
name: classification
effort: low
repeats: 3            # optional; samples per case

system: |
  You classify inbound support messages. Reply with exactly one label.

variants:             # optional alternative prompts, for `evals compare`
  terse: |
    Classify the message. One word.

cases:
  - id: card-declined
    input: My card was declined but I was still charged twice.
    graders:
      - type: exact
        value: billing
```

## Graders

| Type | Checks | Deterministic |
|---|---|---|
| `exact` | Normalised string equality | yes |
| `contains` | All required terms present, no forbidden ones (partial credit) | yes |
| `regex` | Pattern matches the output | yes |
| `json_fields` | Output parses as JSON and fields match (partial credit) | yes |
| `llm_judge` | A judge model scores 1–5 against natural-language criteria | no |

Prefer the deterministic four. `llm_judge` costs an extra API call per trial and
adds its own variance — use it only for genuinely free-form output. It uses
structured outputs, so its score is always a validated integer, never scraped
out of prose.

## Statistics

- **Wilson score intervals** rather than the normal approximation: eval suites
  are small and pass rates sit near 1.0, exactly where the naive interval runs
  past 100% or collapses to zero width at a perfect score.
- **Two-proportion z-test** for regressions. A 6/10 run against a 90% baseline
  of 10 trials looks alarming and is statistically meaningless; the gate says so
  instead of going red.
- **Power analysis** (`evals power`) turns "is this suite big enough?" into a
  number. Detecting a 10-point drop from a 90% baseline takes ~199 trials per
  arm — worth knowing before trusting a six-case suite.

Everything is closed-form, so there is no scipy dependency.

## Cost control

- **Response cache** (`.eval-cache/`) is content-addressed on model, effort,
  system prompt, input, `max_tokens`, and repeat index. Editing a prompt misses
  by construction, so a hit is always legitimate. CI restores it between runs,
  so a PR pays only for what it changed.
- **Prompt caching** on the API side: each suite's system prompt carries a cache
  breakpoint and is sent first, so the prefix is bought once per suite.
- **Budget gate**: a run costing more than `EVAL_BUDGET_USD` ($5 default) fails
  even with perfect scores.

## The baseline

`baselines/main.json` holds the pass rate, trial count, and successes per suite.
CI rewrites it after a merge to `main` (at higher `repeats`, since it is compared
against for weeks) and the bot commits it. Do not edit it by hand — a PR that
needs a lower bar is a PR that made things worse.

Regenerate it deliberately when you change the model or a suite's contents;
scores are not comparable across either.

## Configuration

Everything below is an env var, with defaults in `evals/config.py`.

| Variable | Default | Meaning |
|---|---|---|
| `EVAL_MODEL` | `claude-opus-5` | Model under test |
| `EVAL_JUDGE_MODEL` | `claude-opus-5` | Model used by `llm_judge` |
| `EVAL_EFFORT` | `medium` | `low` … `max` |
| `EVAL_REPEATS` | `1` | Samples per case |
| `EVAL_ALPHA` | `0.05` | Significance level |
| `EVAL_MIN_PASS_RATE` | `0.80` | Absolute floor |
| `EVAL_MAX_REGRESSION` | `0.05` | Allowed drop vs baseline |
| `EVAL_REQUIRE_SIGNIFICANCE` | `1` | Require p < alpha to fail on regression |
| `EVAL_FAIL_ON_FLAKY` | `0` | Fail when a case flips across repeats |
| `EVAL_BUDGET_USD` | `5.00` | Per-run cost ceiling |
| `EVAL_CONCURRENCY` | `8` | Parallel trials |
| `EVAL_CACHE` | `1` | Set `0` to bypass the response cache |
| `EVAL_CACHE_TTL` | `1209600` | Cache entry lifetime, seconds |
