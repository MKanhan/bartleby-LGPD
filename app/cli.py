"""bartleby — LGPD gate for AI agents (free, open-core). Runs `bartleby check`.

This is the check-only CLI of the public gate package. The gate logic lives in `app/cli_check.py`,
shared verbatim with Bartleby's paid on-prem CLI, so the check never drifts. The paid `analyze`
command (RIPD/ROPA/Risk-Map generation) is not part of this package — see https://bartleby.com.br.
"""

from __future__ import annotations

import argparse

from app.cli_check import add_check_subcommand, run_check


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bartleby",
        description="Bartleby — gate de LGPD para agentes de IA (grátis, determinístico, sem LLM).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    add_check_subcommand(sub)

    args = parser.parse_args(argv)
    if args.command == "check":
        return run_check(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
