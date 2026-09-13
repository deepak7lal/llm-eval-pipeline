"""Graders turn a model output into a pass/fail plus a score in [0, 1].

Every grader has the same shape:

    grade(output: str, spec: dict, case: dict) -> GradeResult

`spec` is the grader's own config from the dataset YAML; `case` is the whole
case, so a grader can reach the input when it needs to.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, Field, ValidationError

from . import config


@dataclass
class GradeResult:
    passed: bool
    score: float
    detail: str


def _normalise(text: str, spec: dict) -> str:
    out = text.strip()
    if spec.get("ignore_case", True):
        out = out.lower()
    if spec.get("collapse_whitespace", True):
        out = re.sub(r"\s+", " ", out)
    return out


def exact(output: str, spec: dict, case: dict) -> GradeResult:
    """Output must equal `value` after normalisation."""
    expected = _normalise(str(spec["value"]), spec)
    actual = _normalise(output, spec)
    ok = actual == expected
    return GradeResult(ok, 1.0 if ok else 0.0, "" if ok else f"expected {expected!r}, got {actual!r}")


def contains(output: str, spec: dict, case: dict) -> GradeResult:
    """Output must contain every string in `all_of` and none in `none_of`.

    Scores partially: the fraction of `all_of` terms present, zeroed if any
    forbidden term appears.
    """
    actual = _normalise(output, spec)
    required = [_normalise(str(v), spec) for v in spec.get("all_of", [])]
    forbidden = [_normalise(str(v), spec) for v in spec.get("none_of", [])]

    missing = [t for t in required if t not in actual]
    present_forbidden = [t for t in forbidden if t in actual]

    if present_forbidden:
        return GradeResult(False, 0.0, f"forbidden term present: {present_forbidden}")
    score = 1.0 if not required else (len(required) - len(missing)) / len(required)
    ok = not missing
    return GradeResult(ok, score, "" if ok else f"missing: {missing}")


def regex(output: str, spec: dict, case: dict) -> GradeResult:
    """Output must match `pattern`."""
    flags = 0 if spec.get("case_sensitive") else re.IGNORECASE
    ok = re.search(spec["pattern"], output, flags) is not None
    return GradeResult(ok, 1.0 if ok else 0.0, "" if ok else f"no match for {spec['pattern']!r}")


def json_fields(output: str, spec: dict, case: dict) -> GradeResult:
    """Output must parse as JSON and match every key/value in `equals`.

    Scores as the fraction of expected fields that matched, so a near-miss on a
    ten-field extraction does not read the same as total garbage.
    """
    try:
        # Tolerate a fenced block; the model is not always asked for bare JSON.
        payload = re.sub(r"^```(?:json)?|```$", "", output.strip(), flags=re.MULTILINE).strip()
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        return GradeResult(False, 0.0, f"invalid JSON: {exc}")

    expected: dict = spec.get("equals", {})
    if not expected:
        return GradeResult(True, 1.0, "parsed as JSON")

    mismatches = [k for k, v in expected.items() if data.get(k) != v]
    score = (len(expected) - len(mismatches)) / len(expected)
    ok = not mismatches
    detail = "" if ok else "mismatched fields: " + ", ".join(
        f"{k}={data.get(k)!r} (want {expected[k]!r})" for k in mismatches
    )
    return GradeResult(ok, score, detail)


class _Judgement(BaseModel):
    """Schema the judge model is constrained to return."""

    score: int = Field(ge=1, le=5, description="1 = fails the criteria entirely, 5 = fully satisfies them")
    reasoning: str = Field(description="One or two sentences justifying the score")


_JUDGE_SYSTEM = """You are a strict evaluator of AI assistant outputs.

You are given a task, the criteria the answer must satisfy, and a candidate \
answer. Score the candidate from 1 to 5 against the criteria only - not against \
your own preferred phrasing. Style, length, and formatting are irrelevant unless \
the criteria mention them.

5 = fully satisfies every criterion.
4 = satisfies the criteria with a trivial omission.
3 = satisfies the main criterion but misses a stated detail.
2 = addresses the task but violates a criterion.
1 = does not address the task, or is factually wrong."""

# Appended only for providers without a structured-output guarantee.
_JSON_INSTRUCTION = """

Reply with only this JSON object, no code fence and no commentary:
{"score": <integer 1-5>, "reasoning": "<one or two sentences>"}"""


def llm_judge(output: str, spec: dict, case: dict) -> GradeResult:
    """Score with a judge model against natural-language `criteria`.

    Passes at `min_score` (default 4) on the 1-5 scale.

    On Anthropic the judgement is constrained by structured outputs, so the
    score is a validated integer rather than something parsed out of prose.
    Other providers do not all offer that, so there the judge is asked for bare
    JSON and the result is validated here - same schema, weaker guarantee, and
    an unparseable judgement fails the case rather than passing it silently.
    """
    from . import providers
    from .client import complete

    min_score = int(spec.get("min_score", 4))
    prompt = (
        f"<task>\n{case['input']}\n</task>\n\n"
        f"<criteria>\n{spec['criteria']}\n</criteria>\n\n"
        f"<candidate_answer>\n{output}\n</candidate_answer>"
    )

    if config.PROVIDER == "anthropic":
        response = providers.get_provider()._client.messages.parse(
            model=config.JUDGE_MODEL,
            max_tokens=1024,
            system=[{"type": "text", "text": _JUDGE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": config.JUDGE_EFFORT},
            messages=[{"role": "user", "content": prompt}],
            output_format=_Judgement,
        )
        verdict = response.parsed_output
    else:
        completion = complete(
            system=_JUDGE_SYSTEM + _JSON_INSTRUCTION,
            user=prompt,
            model=config.JUDGE_MODEL,
            effort=config.JUDGE_EFFORT,
            max_tokens=1024,
        )
        try:
            payload = re.sub(r"^```(?:json)?|```$", "", completion.text.strip(), flags=re.MULTILINE)
            verdict = _Judgement.model_validate_json(payload.strip())
        except (ValidationError, ValueError) as exc:
            return GradeResult(False, 0.0, f"judge returned unusable output: {exc}")

    ok = verdict.score >= min_score
    return GradeResult(ok, (verdict.score - 1) / 4, f"judge {verdict.score}/5: {verdict.reasoning}")


GRADERS: dict[str, Callable[[str, dict, dict], GradeResult]] = {
    "exact": exact,
    "contains": contains,
    "regex": regex,
    "json_fields": json_fields,
    "llm_judge": llm_judge,
}


def grade(output: str, spec: dict, case: dict) -> GradeResult:
    """Dispatch to the grader named by `spec['type']`."""
    kind = spec.get("type")
    if kind not in GRADERS:
        raise ValueError(f"unknown grader {kind!r}; known: {sorted(GRADERS)}")
    return GRADERS[kind](output, spec, case)
