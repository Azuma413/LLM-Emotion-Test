from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from llm_emotion_test.config import ConfigError, config_summary, load_config


COMMANDS = {
    "prepare-data": "Load and validate data preparation settings.",
    "train-sft": "Load and validate SFT training settings.",
    "distill": "Load and validate distillation settings.",
    "train-rl": "Load and validate GRPO training settings.",
    "evaluate": "Load and validate evaluation settings.",
    "sample-dialogue": "Load and validate dialogue sampling settings.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-emotion-test",
        description="Research pipeline CLI for latent emotion experiments.",
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to a YAML config file. Defaults to configs/base.yaml.",
    )

    subparsers = parser.add_subparsers(dest="command")
    for name, help_text in COMMANDS.items():
        subparser = subparsers.add_parser(name, help=help_text, description=help_text)
        subparser.add_argument(
            "--config",
            default=None,
            help="Path to a YAML config file. Overrides the global --config option.",
        )
        subparser.set_defaults(command=name)

    return parser


def run_command(command: str, config_path: str | Path, console: Console) -> int:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 2

    table = Table(title=f"{command} configuration")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    for key, value in config_summary(config).items():
        table.add_row(key, str(value))

    console.print(table)
    console.print(
        "[yellow]Pipeline implementation for this command starts in the next phase.[/yellow]"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    config_path = args.config or "configs/base.yaml"
    return run_command(args.command, config_path, Console())


if __name__ == "__main__":
    raise SystemExit(main())
