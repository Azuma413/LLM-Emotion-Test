from __future__ import annotations

import csv
import json

from llm_emotion_test.config import EvaluationConfig, ExperimentConfig, OutputConfig, RLTaskConfig
from llm_emotion_test.evaluation.evaluator import (
    entropy,
    mutual_information,
    run_evaluation,
)


def test_entropy_and_mutual_information_are_non_negative() -> None:
    assert entropy([0, 0, 1, 1]) > 0.0
    assert mutual_information([(0, 0), (0, 0), (1, 1), (1, 1)]) > 0.0
    assert mutual_information([]) == 0.0


def test_run_evaluation_writes_metrics_report_and_visualizations(tmp_path) -> None:
    config = ExperimentConfig(
        rl_task=RLTaskConfig(
            policy_backend="tabular_smoke",
            max_generation_attempts=500,
            max_turns=3,
            eval_episodes=1,
        ),
        evaluation=EvaluationConfig(
            num_tasks=2,
            variants=["latent_fixed", "latent_random", "no_latent"],
            transcript_sample_count=1,
            failure_sample_count=1,
        ),
        output=OutputConfig(root=tmp_path, run_id="eval"),
        training={"report_to": "none"},
    )

    metrics = run_evaluation(config)

    assert metrics["num_variants"] == 3
    assert metrics["num_tasks"] == 2
    assert (config.output.run_dir / "evaluation_metrics.csv").exists()
    assert (config.output.run_dir / "evaluation_report.md").exists()
    assert (config.output.run_dir / "evaluation_transcripts.jsonl").exists()
    assert (config.output.run_dir / "latent_transition_heatmap.svg").exists()
    assert (config.output.run_dir / "reward_curve.svg").exists()
    assert (config.output.run_dir / "emotion_distribution.svg").exists()

    with (config.output.run_dir / "evaluation_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))
    assert {row["variant"] for row in rows} == {
        "latent_fixed",
        "latent_random",
        "no_latent",
    }
    no_latent = next(row for row in rows if row["variant"] == "no_latent")
    assert float(no_latent["parser_failure_rate"]) == 1.0

    with config.output.metrics_path.open(encoding="utf-8") as file:
        summary_records = [json.loads(line) for line in file if line.strip()]
    assert len(summary_records) == 3
