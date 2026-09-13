"""Command line entry point: `python -m evals [subcommand]`.

Subcommands:
  run       (default) score every suite and apply the gate
  compare   A/B two prompt variants with a significance test
  power     how many trials a suite needs to detect a given regression
  cache     inspect or clear the response cache
"""

from __future__ import annotations

import argparse
import sys

from . import cache as cache_mod
from . import compare as compare_mod
from . import config
from .gate import ContaminatedBaseline, evaluate, write_baseline
from .report import to_markdown, write_reports
from .runner import discover_suites, run_all
from .stats import required_trials


def _apply_overrides(args: argparse.Namespace) -> None:
    """Config is read at import time, so CLI overrides patch the module."""
    if getattr(args, "model", None):
        config.MODEL = args.model
    if getattr(args, "effort", None):
        config.EFFORT = args.effort
    if getattr(args, "repeats", None):
        config.REPEATS = args.repeats


def cmd_run(args: argparse.Namespace) -> int:
    _apply_overrides(args)

    results = run_all(only=args.suites, repeats=args.repeats, use_cache=not args.no_cache)
    if not results:
        print("no suites found", file=sys.stderr)
        return 2

    run = evaluate(results)
    md_path, json_path = write_reports(results, run)

    print(to_markdown(results, run))
    print(f"wrote {md_path} and {json_path}", file=sys.stderr)

    if args.update_baseline:
        try:
            path = write_baseline(results)
        except ContaminatedBaseline as exc:
            print(f"baseline NOT updated: {exc}", file=sys.stderr)
            return 1
        print(f"baseline updated: {path}", file=sys.stderr)

    if args.no_gate:
        return 0
    return 1 if run.failed else 0


def cmd_compare(args: argparse.Namespace) -> int:
    _apply_overrides(args)

    suites = discover_suites(only=args.suites)
    comparisons = []
    for suite in suites:
        available = (suite.get("variants") or {}).keys()
        if args.variant not in available:
            print(
                f"skipping `{suite['name']}`: no variant {args.variant!r} "
                f"(has: {', '.join(available) or 'none'})",
                file=sys.stderr,
            )
            continue
        comparisons.append(
            compare_mod.compare(
                suite,
                treatment=args.variant,
                control=args.control,
                repeats=args.repeats,
                use_cache=not args.no_cache,
            )
        )

    if not comparisons:
        print(f"no suite declares a variant named {args.variant!r}", file=sys.stderr)
        return 2

    print(compare_mod.to_markdown(comparisons))

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / "comparison.md"
    out.write_text(compare_mod.to_markdown(comparisons), encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)

    # A comparison is a research tool, not a gate - it never fails the build.
    return 0


def cmd_power(args: argparse.Namespace) -> int:
    """Answer 'is this suite big enough to catch what I care about?'"""
    needed = required_trials(effect=args.effect, base_rate=args.base_rate, alpha=config.ALPHA)
    print(
        f"To detect a {args.effect:.0%} drop from a {args.base_rate:.0%} baseline "
        f"(alpha={config.ALPHA}, power=80%): {needed} trials per arm."
    )
    for suite in discover_suites(only=args.suites):
        cases = len(suite["cases"])
        repeats = -(-needed // cases)  # ceiling division
        print(
            f"  `{suite['name']}`: {cases} cases -> "
            f"{repeats} repeat(s) per case ({cases * repeats} trials)"
        )
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    if args.clear:
        removed = cache_mod.clear()
        print(f"removed {removed} cache entries")
        return 0
    stats = cache_mod.stats()
    state = "enabled" if cache_mod.ENABLED else "disabled (EVAL_CACHE=0)"
    print(
        f"cache {state} at {cache_mod.CACHE_DIR}\n"
        f"  {stats['entries']} entries, {stats['bytes'] / 1024:.1f} KiB\n"
        f"  TTL {cache_mod.TTL_SECONDS // 3600}h"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evals", description="Run the Claude eval suites.")
    sub = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--suite", action="append", dest="suites", help="limit to this suite (repeatable)")
        p.add_argument("--model", help=f"override the model under test (default {config.MODEL})")
        p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
        p.add_argument(
            "--repeats", type=int,
            help=f"samples per case (default {config.REPEATS}; higher tightens the interval)",
        )
        p.add_argument("--no-cache", action="store_true", help="ignore the on-disk response cache")

    run = sub.add_parser("run", help="score every suite and apply the gate")
    add_common(run)
    run.add_argument(
        "--update-baseline",
        action="store_true",
        help="overwrite baselines/main.json with this run - only do this on main",
    )
    run.add_argument("--no-gate", action="store_true", help="report scores but always exit 0")
    run.set_defaults(func=cmd_run)

    comp = sub.add_parser("compare", help="A/B a prompt variant against the baseline prompt")
    add_common(comp)
    comp.add_argument("--variant", required=True, help="name of the variant to test")
    comp.add_argument("--control", default="baseline", help="arm to compare against")
    comp.set_defaults(func=cmd_compare)

    power = sub.add_parser("power", help="trials needed to detect a regression of a given size")
    power.add_argument("--suite", action="append", dest="suites")
    power.add_argument("--effect", type=float, default=0.10, help="drop to detect (default 0.10)")
    power.add_argument("--base-rate", type=float, default=0.90, help="assumed baseline pass rate")
    power.set_defaults(func=cmd_power)

    cache_p = sub.add_parser("cache", help="inspect or clear the response cache")
    cache_p.add_argument("--clear", action="store_true")
    cache_p.set_defaults(func=cmd_cache)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()

    # Bare `python -m evals` and `python -m evals --suite x` both mean "run",
    # but -h/--help must still reach the top-level parser.
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv = ["run", *argv]

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
