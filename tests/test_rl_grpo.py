from __future__ import annotations

import json

from llm_emotion_test.config import ExperimentConfig, OutputConfig, RLTaskConfig
from llm_emotion_test.training.rl import (
    RolloutStep,
    assign_at_grpo_advantages,
    train_at_grpo_smoke,
)


def test_at_grpo_advantages_are_turn_group_normalized() -> None:
    rollouts = [
        {
            "steps": [
                RolloutStep(
                    prompt="p",
                    generated_text="a",
                    logprob=-1.0,
                    agent_id="A",
                    latent_id=0,
                    previous_latent_id=0,
                    reward=0.0,
                    team_reward=0.0,
                    local_reward=0.0,
                    advantage=0.0,
                    group_id="g",
                    episode_index=0,
                    rollout_index=0,
                    turn_index=0,
                )
            ]
        },
        {
            "steps": [
                RolloutStep(
                    prompt="p",
                    generated_text="b",
                    logprob=-1.0,
                    agent_id="A",
                    latent_id=1,
                    previous_latent_id=0,
                    reward=2.0,
                    team_reward=1.0,
                    local_reward=1.0,
                    advantage=0.0,
                    group_id="g",
                    episode_index=0,
                    rollout_index=1,
                    turn_index=0,
                )
            ]
        },
    ]

    assign_at_grpo_advantages(rollouts, epsilon=1e-6)

    assert rollouts[0]["steps"][0].advantage < 0.0
    assert rollouts[1]["steps"][0].advantage > 0.0


def test_train_at_grpo_smoke_writes_outputs(tmp_path) -> None:
    config = ExperimentConfig(
        rl_task=RLTaskConfig(
            max_generation_attempts=500,
            num_episodes=1,
            rollouts_per_problem=2,
            max_turns=3,
            eval_episodes=1,
        ),
        output=OutputConfig(root=tmp_path, run_id="rl"),
        training={"report_to": "none"},
    )

    metrics = train_at_grpo_smoke(config)

    assert metrics["mode"] == "at_grpo_smoke"
    assert metrics["sampling_scheme"] == "tree-structured per-agent-turn branching"
    assert metrics["num_rollouts"] == 1
    assert metrics["num_rollout_steps"] >= 2
    assert metrics["num_branch_samples"] == metrics["num_rollout_steps"]
    assert metrics["num_branch_groups"] >= 1
    assert "fixed_baseline_success_rate" in metrics
    assert "latent_usage_distribution" in metrics
    assert (config.output.run_dir / config.rl_task.transcript_filename).exists()
    assert (config.output.run_dir / "rollout_buffer.jsonl").exists()
    checkpoint_path = config.output.checkpoints_dir / config.rl_task.checkpoint_filename
    assert checkpoint_path.exists()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["episode_index"] == 1

    with (config.output.run_dir / "rollout_buffer.jsonl").open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    group_counts: dict[str, int] = {}
    for record in records:
        group_counts[record["group_id"]] = group_counts.get(record["group_id"], 0) + 1
    assert set(group_counts.values()) == {config.rl_task.rollouts_per_problem}
