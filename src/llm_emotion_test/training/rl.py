from __future__ import annotations

import statistics
from typing import Any

import yaml

from llm_emotion_test.agents.negotiation import RuleBasedConstraintSharingAgent
from llm_emotion_test.config import ExperimentConfig
from llm_emotion_test.tasks.negotiation_env import (
    CooperativeHiddenConstraintsEnv,
    write_transcripts,
)
from llm_emotion_test.training.sft import write_jsonl


def run_rule_based_rl_smoke(config: ExperimentConfig) -> dict[str, Any]:
    config.output.run_dir.mkdir(parents=True, exist_ok=True)
    config.output.config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    agents = {
        "A": RuleBasedConstraintSharingAgent(latent_id=0),
        "B": RuleBasedConstraintSharingAgent(latent_id=0),
    }
    transcript_records = []
    for episode_index in range(config.rl_task.num_episodes):
        env = CooperativeHiddenConstraintsEnv(config)
        env.reset(seed=config.runtime.seed + episode_index)
        while not env.is_done:
            agent_id = env._require_state().active_agent_id
            observation = env.observe(agent_id)
            action = agents[agent_id].act(observation)
            env.step(action)
        record = env.transcript_record()
        record["episode_index"] = episode_index
        transcript_records.append(record)

    transcript_path = config.output.run_dir / config.rl_task.transcript_filename
    write_transcripts(transcript_records, transcript_path)

    rewards = [float(record["total_reward"]) for record in transcript_records]
    successes = [bool(record["success"]) for record in transcript_records]
    metrics = {
        "mode": "rule_based_smoke",
        "num_episodes": len(transcript_records),
        "success_rate": sum(successes) / len(successes) if successes else 0.0,
        "mean_total_reward": statistics.fmean(rewards) if rewards else 0.0,
        "transcript_path": str(transcript_path),
    }
    write_jsonl([metrics], config.output.metrics_path)
    return metrics
