#!/usr/bin/env python3
"""Isolated explicit-template CLI for registered baseline diagnostics only."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import run_benchmark as formal_cli
from eval.gsa_eval import template_diagnostic


def _extract_template_type(argv: list[str], option: str = "--template-type") -> tuple[str | None, list[str]]:
    template_type = None
    cleaned = [argv[0]]
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == option:
            if template_type is not None or index + 1 >= len(argv):
                raise SystemExit(f"{option} must appear exactly once with a value")
            template_type = argv[index + 1]
            index += 2
            continue
        if arg.startswith(option + "="):
            if template_type is not None:
                raise SystemExit(f"{option} must appear at most once")
            template_type = arg.split("=", 1)[1]
            index += 1
            continue
        cleaned.append(arg)
        index += 1
    return template_type, cleaned


def main() -> int:
    # Reuse the strict option parser; leave the frozen benchmark CLI unchanged.
    model_type, model_cleaned = _extract_template_type(sys.argv, "--model-type")
    template_type, cleaned_argv = _extract_template_type(model_cleaned)
    try:
        template_diagnostic.configure_template_type(template_type, model_type)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    formal_cli.generate_shard_multi = template_diagnostic.generate_shard_multi
    formal_cli.evaluate = template_diagnostic.evaluate
    formal_cli.score_shards = template_diagnostic.score_shards
    sys.argv = cleaned_argv
    return formal_cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
