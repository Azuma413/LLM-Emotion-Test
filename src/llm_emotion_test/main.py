from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from llm_emotion_test.config import ConfigError, config_summary, load_config
from llm_emotion_test.data.wrime import prepare_wrime_dataset
from llm_emotion_test.evaluation.evaluator import run_evaluation
from llm_emotion_test.training.distill import train_distill
from llm_emotion_test.training.rl import run_rule_based_rl_smoke
from llm_emotion_test.training.sft import train_sft


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
    if command == "prepare-data":
        return run_prepare_data(config_path, console)
    if command == "train-sft":
        return run_train_sft(config_path, console)
    if command == "distill":
        return run_distill(config_path, console)
    if command == "train-rl":
        return run_train_rl(config_path, console)
    if command == "evaluate":
        return run_evaluate(config_path, console)

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


def run_train_sft(config_path: str | Path, console: Console) -> int:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 2

    try:
        metrics = train_sft(config)
    except Exception as exc:
        console.print(f"[bold red]SFT training failed:[/bold red] {exc}")
        return 1

    table = Table(title="train-sft summary")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_row("final_checkpoint", str(metrics["final_checkpoint"]))
    table.add_row("eval_loss", str(metrics["eval"].get("eval_loss")))
    table.add_row(
        "sample_latent_marker_accuracy",
        str(metrics["sample_latent_marker_accuracy"]),
    )
    console.print(table)
    return 0


def run_distill(config_path: str | Path, console: Console) -> int:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 2

    try:
        metrics = train_distill(config)
    except Exception as exc:
        console.print(f"[bold red]Distillation failed:[/bold red] {exc}")
        return 1

    distill_stats = metrics["distillation"]
    table = Table(title="distill summary")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_row("distill_data", str(distill_stats["output_path"]))
    table.add_row("student_data", str(distill_stats["student_data_path"]))
    table.add_row("teacher_cache", str(distill_stats["teacher_cache_path"]))
    table.add_row("num_distill_records", str(distill_stats["num_distill_records"]))
    table.add_row("final_checkpoint", str(metrics["final_checkpoint"]))
    table.add_row("eval_loss", str(metrics["eval"].get("eval_loss")))
    table.add_row(
        "sample_latent_marker_accuracy",
        str(metrics["sample_latent_marker_accuracy"]),
    )
    console.print(table)
    return 0


def run_train_rl(config_path: str | Path, console: Console) -> int:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 2

    try:
        metrics = run_rule_based_rl_smoke(config)
    except Exception as exc:
        console.print(f"[bold red]RL smoke run failed:[/bold red] {exc}")
        return 1

    table = Table(title="train-rl summary")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_row("mode", str(metrics["mode"]))
    table.add_row("num_episodes", str(metrics["num_episodes"]))
    table.add_row("success_rate", str(metrics["success_rate"]))
    table.add_row("mean_total_reward", str(metrics["mean_total_reward"]))
    if "num_rollout_steps" in metrics:
        table.add_row("num_rollout_steps", str(metrics["num_rollout_steps"]))
    if "latent_usage_entropy" in metrics:
        table.add_row("latent_usage_entropy", str(metrics["latent_usage_entropy"]))
    if "parser_failure_rate" in metrics:
        table.add_row("parser_failure_rate", str(metrics["parser_failure_rate"]))
    table.add_row("transcript_path", str(metrics["transcript_path"]))
    if "rollout_buffer_path" in metrics:
        table.add_row("rollout_buffer_path", str(metrics["rollout_buffer_path"]))
    if "checkpoint_path" in metrics:
        table.add_row("checkpoint_path", str(metrics["checkpoint_path"]))
    console.print(table)
    return 0


def run_evaluate(config_path: str | Path, console: Console) -> int:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 2

    try:
        metrics = run_evaluation(config)
    except Exception as exc:
        console.print(f"[bold red]Evaluation failed:[/bold red] {exc}")
        return 1

    table = Table(title="evaluate summary")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_row("num_variants", str(metrics["num_variants"]))
    table.add_row("num_tasks", str(metrics["num_tasks"]))
    table.add_row("metrics_csv", str(metrics["metrics_csv_path"]))
    table.add_row("transcripts", str(metrics["transcript_path"]))
    table.add_row("report", str(metrics["report_path"]))
    console.print(table)
    return 0


def run_prepare_data(config_path: str | Path, console: Console) -> int:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 2

    try:
        stats = prepare_wrime_dataset(config)
    except Exception as exc:
        console.print(f"[bold red]Data preparation failed:[/bold red] {exc}")
        return 1

    summary = Table(title="prepare-data summary")
    summary.add_column("Key", style="cyan")
    summary.add_column("Value")
    summary.add_row("output_path", str(stats["output_path"]))
    summary.add_row("num_samples", str(stats["num_samples"]))
    summary.add_row("text_length", str(stats["text_length"]))
    console.print(summary)

    for key in (
        "split_counts",
        "label_counts",
        "input_latent_counts",
        "target_latent_counts",
    ):
        table = Table(title=key)
        table.add_column("Value", style="cyan")
        table.add_column("Count")
        for value, count in stats[key].items():
            table.add_row(str(value), str(count))
        console.print(table)

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
