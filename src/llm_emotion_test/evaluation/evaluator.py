from __future__ import annotations

import csv
import math
import random
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from llm_emotion_test.agents.negotiation import (
    AgentAction,
    LLMNegotiationAgent,
    RuleBasedConstraintSharingAgent,
)
from llm_emotion_test.config import ExperimentConfig
from llm_emotion_test.models.loader import (
    build_soft_prompt_model,
    load_soft_prompt_model_from_checkpoint,
)
from llm_emotion_test.tasks.negotiation_env import CooperativeHiddenConstraintsEnv, write_transcripts
from llm_emotion_test.training.rl import (
    TabularLatentPolicy,
    load_or_create_policy,
    run_policy_rollout,
)
from llm_emotion_test.training.sft import write_jsonl


class SingleConstraintAgent:
    def __init__(self, *, latent_id: int = 0) -> None:
        self.latent_id = latent_id

    def act(self, observation: Mapping[str, Any]) -> AgentAction:
        private_constraints = observation["private_constraints"]
        transcript = observation["transcript"]
        disclosed = {
            turn["action"]["message_text"]
            for turn in transcript
            if turn["agent_id"] == observation["agent_id"]
        }
        message = ""
        for constraint in private_constraints:
            if constraint.text not in disclosed:
                message = constraint.text
                break
        if not message:
            message = "共有済みの制約から候補を確認しましょう。"
        return AgentAction(
            message_text=message,
            next_latent_id=self.latent_id,
            raw_text=message,
        )


class AlternatingLatentAgent:
    def __init__(self, *, num_latents: int) -> None:
        self.num_latents = num_latents
        self.base_agent = RuleBasedConstraintSharingAgent(latent_id=0)

    def act(self, observation: Mapping[str, Any]) -> AgentAction:
        action = self.base_agent.act(observation)
        latent_id = (
            int(observation["turn_index"])
            + (0 if observation["agent_id"] == "A" else 1)
        ) % self.num_latents
        return AgentAction(
            message_text=action.message_text,
            next_latent_id=latent_id,
            proposal=action.proposal,
            raw_text=action.raw_text,
            parse_error=action.parse_error,
        )


class RandomLatentAgent:
    def __init__(self, *, num_latents: int, rng: random.Random) -> None:
        self.num_latents = num_latents
        self.rng = rng
        self.base_agent = RuleBasedConstraintSharingAgent(latent_id=0)

    def act(self, observation: Mapping[str, Any]) -> AgentAction:
        action = self.base_agent.act(observation)
        return AgentAction(
            message_text=action.message_text,
            next_latent_id=self.rng.randrange(self.num_latents),
            proposal=action.proposal,
            raw_text=action.raw_text,
            parse_error=action.parse_error,
        )


class NoLatentAgent:
    def __init__(self) -> None:
        self.base_agent = RuleBasedConstraintSharingAgent(latent_id=0)

    def act(self, observation: Mapping[str, Any]) -> AgentAction:
        action = self.base_agent.act(observation)
        return AgentAction(
            message_text=action.message_text,
            next_latent_id=0,
            proposal=action.proposal,
            raw_text=action.raw_text,
            parse_error="latent disabled by ablation",
        )


def run_evaluation(config: ExperimentConfig) -> dict[str, Any]:
    config.output.run_dir.mkdir(parents=True, exist_ok=True)
    config.output.config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    all_transcripts: list[dict[str, Any]] = []
    comparison_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    model_cache: dict[str, tuple[Any, Any]] = {}
    seeds = [
        config.runtime.seed + config.evaluation.task_seed_offset + index
        for index in range(config.evaluation.num_tasks)
    ]

    for variant in config.evaluation.variants:
        transcripts = [
            run_variant_episode(
                config,
                variant=variant,
                episode_index=index,
                problem_seed=seed,
                model_cache=model_cache,
            )
            for index, seed in enumerate(seeds)
        ]
        all_transcripts.extend(transcripts)
        summaries.append(summarize_variant(variant, transcripts))
        comparison_records.extend(build_comparison_records(variant, transcripts))

    transcript_path = config.output.run_dir / config.evaluation.transcript_filename
    comparison_path = config.output.run_dir / config.evaluation.comparison_filename
    metrics_csv_path = config.output.run_dir / config.evaluation.metrics_csv_filename
    report_path = config.output.run_dir / config.evaluation.report_filename
    heatmap_path = config.output.run_dir / config.evaluation.latent_heatmap_filename
    reward_curve_path = config.output.run_dir / config.evaluation.reward_curve_filename
    emotion_distribution_path = (
        config.output.run_dir / config.evaluation.emotion_distribution_filename
    )

    write_transcripts(all_transcripts, transcript_path)
    write_jsonl(comparison_records, comparison_path)
    write_metrics_csv(summaries, metrics_csv_path)
    write_jsonl(summaries, config.output.metrics_path)
    write_latent_heatmap(all_transcripts, heatmap_path)
    write_reward_curve(all_transcripts, reward_curve_path)
    write_emotion_distribution(all_transcripts, emotion_distribution_path)
    write_report(
        summaries,
        all_transcripts,
        report_path,
        metrics_csv_path=metrics_csv_path,
        transcript_path=transcript_path,
        heatmap_path=heatmap_path,
        reward_curve_path=reward_curve_path,
        emotion_distribution_path=emotion_distribution_path,
        transcript_sample_count=config.evaluation.transcript_sample_count,
        failure_sample_count=config.evaluation.failure_sample_count,
    )

    return {
        "num_variants": len(summaries),
        "num_tasks": config.evaluation.num_tasks,
        "metrics_path": str(config.output.metrics_path),
        "metrics_csv_path": str(metrics_csv_path),
        "transcript_path": str(transcript_path),
        "comparison_path": str(comparison_path),
        "report_path": str(report_path),
        "latent_heatmap_path": str(heatmap_path),
        "reward_curve_path": str(reward_curve_path),
        "emotion_distribution_path": str(emotion_distribution_path),
        "summaries": summaries,
    }


def run_variant_episode(
    config: ExperimentConfig,
    *,
    variant: str,
    episode_index: int,
    problem_seed: int,
    model_cache: dict[str, tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    model_cache = model_cache if model_cache is not None else {}
    llm_agents = build_llm_variant_agents(
        config,
        variant=variant,
        model_cache=model_cache,
    )
    if llm_agents is not None:
        env = CooperativeHiddenConstraintsEnv(config)
        env.reset(seed=problem_seed)
        while not env.is_done:
            agent_id = env._require_state().active_agent_id
            env.step(llm_agents[agent_id].act(env.observe(agent_id)))
        transcript = env.transcript_record()
        transcript["episode_index"] = episode_index
        transcript["variant"] = variant
        transcript["policy_source"] = "llm_checkpoint"
        return transcript

    if variant == "rl_model":
        policy = load_evaluation_policy(config, rng=random.Random(problem_seed))
        if policy is not None:
            transcript = run_policy_rollout(
                config,
                policy,
                episode_index=episode_index,
                rollout_index=0,
                rollout_group_id=f"eval-{variant}-{episode_index}",
                problem_seed=problem_seed,
            )["transcript"]
            transcript["variant"] = variant
            transcript["policy_source"] = "tabular_checkpoint"
            return transcript

    env = CooperativeHiddenConstraintsEnv(config)
    env.reset(seed=problem_seed)
    agents = build_variant_agents(config, variant=variant, seed=problem_seed)
    while not env.is_done:
        agent_id = env._require_state().active_agent_id
        env.step(agents[agent_id].act(env.observe(agent_id)))
    transcript = env.transcript_record()
    transcript["episode_index"] = episode_index
    transcript["variant"] = variant
    transcript["policy_source"] = "rule_based_surrogate"
    return transcript


def build_llm_variant_agents(
    config: ExperimentConfig,
    *,
    variant: str,
    model_cache: dict[str, tuple[Any, Any]],
) -> dict[str, LLMNegotiationAgent] | None:
    checkpoint_dir = checkpoint_dir_for_variant(config, variant)
    if config.evaluation.model_variant_backend == "surrogate":
        return None
    if checkpoint_dir is None:
        if not (variant == "base_model" and config.evaluation.model_variant_backend == "llm"):
            return None
    elif not checkpoint_dir.exists():
        if config.evaluation.model_variant_backend == "llm":
            raise FileNotFoundError(f"Checkpoint for {variant} does not exist: {checkpoint_dir}")
        return None

    cache_key = f"{variant}:{checkpoint_dir or config.model.base_model}"
    if cache_key not in model_cache:
        if checkpoint_dir is None:
            model_cache[cache_key] = build_soft_prompt_model(config)
        else:
            model_cache[cache_key] = load_soft_prompt_model_from_checkpoint(
                checkpoint_dir,
                config,
            )
    model, tokenizer = model_cache[cache_key]
    return {
        agent_id: LLMNegotiationAgent(
            model,
            tokenizer,
            agent_id=agent_id,
            num_latents=config.soft_prompt.num_latents,
            marker_template=config.soft_prompt.latent_marker_template,
            anchor_token=config.latent_training.anchor_token,
            generation_max_new_tokens=config.training.generation_max_new_tokens,
            fallback=config.soft_prompt.invalid_latent_fallback,
            neutral_latent_id=config.soft_prompt.neutral_latent_id,
        )
        for agent_id in ("A", "B")
    }


def checkpoint_dir_for_variant(config: ExperimentConfig, variant: str) -> Path | None:
    if variant == "base_model":
        return config.evaluation.base_model_checkpoint_dir
    if variant == "sft_model":
        return config.evaluation.sft_model_checkpoint_dir
    if variant == "distilled_model":
        return config.evaluation.distilled_model_checkpoint_dir
    if variant == "rl_model":
        return config.evaluation.rl_model_checkpoint_dir
    return None


def load_evaluation_policy(
    config: ExperimentConfig,
    *,
    rng: random.Random,
) -> TabularLatentPolicy | None:
    checkpoint_path = None
    if config.rl_task.resume_from_checkpoint is not None:
        checkpoint_path = config.rl_task.resume_from_checkpoint
    elif config.evaluation.source_run_dir is not None:
        checkpoint_path = (
            config.evaluation.source_run_dir
            / "checkpoints"
            / config.rl_task.checkpoint_filename
        )
    if checkpoint_path is None or not checkpoint_path.exists():
        return None

    policy_config = config.model_copy(
        update={"rl_task": config.rl_task.model_copy(update={"resume_from_checkpoint": checkpoint_path})}
    )
    policy, _ = load_or_create_policy(policy_config, rng=rng)
    return policy


def build_variant_agents(
    config: ExperimentConfig,
    *,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    if variant == "base_model":
        return {agent_id: SingleConstraintAgent(latent_id=0) for agent_id in ("A", "B")}
    if variant in {"sft_model", "latent_fixed"}:
        return {
            agent_id: RuleBasedConstraintSharingAgent(latent_id=0)
            for agent_id in ("A", "B")
        }
    if variant == "distilled_model":
        return {
            agent_id: AlternatingLatentAgent(num_latents=config.soft_prompt.num_latents)
            for agent_id in ("A", "B")
        }
    if variant == "latent_random":
        return {
            agent_id: RandomLatentAgent(
                num_latents=config.soft_prompt.num_latents,
                rng=random.Random(seed + index),
            )
            for index, agent_id in enumerate(("A", "B"))
        }
    if variant == "no_latent":
        return {agent_id: NoLatentAgent() for agent_id in ("A", "B")}
    return {
        agent_id: RuleBasedConstraintSharingAgent(latent_id=0)
        for agent_id in ("A", "B")
    }


def summarize_variant(variant: str, transcripts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rewards = [float(record["total_reward"]) for record in transcripts]
    successes = [bool(record["success"]) for record in transcripts]
    actions = [
        action
        for record in transcripts
        for action in (item["action"] for item in record["transcript"])
    ]
    utterance_lengths = [len(str(action.get("message_text", ""))) for action in actions]
    parser_failures = [action.get("parse_error") is not None for action in actions]
    latent_ids = [int(action.get("next_latent_id", 0)) for action in actions]
    transitions = [
        (previous, current)
        for record in transcripts
        for previous, current in latent_transitions(record)
    ]
    task_states = [
        (int(item["turn_index"]), int(item["action"].get("next_latent_id", 0)))
        for record in transcripts
        for item in record["transcript"]
    ]
    emotion_counts = Counter(classify_emotion(action.get("message_text", "")) for action in actions)
    return {
        "variant": variant,
        "num_tasks": len(transcripts),
        "mean_reward": mean_or_zero(rewards),
        "agreement_rate": sum(successes) / len(successes) if successes else 0.0,
        "pareto_efficiency": pareto_efficiency(transcripts),
        "fairness": fairness_score(transcripts),
        "mean_utterance_length": mean_or_zero(utterance_lengths),
        "parser_failure_rate": (
            sum(parser_failures) / len(parser_failures) if parser_failures else 0.0
        ),
        "latent_usage_entropy": entropy(latent_ids),
        "latent_transition_entropy": entropy(transitions),
        "task_state_latent_mutual_information": mutual_information(task_states),
        "emotion_distribution": dict(sorted(emotion_counts.items())),
        "reward_trend": rewards,
    }


def build_comparison_records(
    variant: str,
    transcripts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "variant": variant,
            "episode_index": record.get("episode_index", index),
            "answer": record["task"]["answer"],
            "success": bool(record["success"]),
            "total_reward": float(record["total_reward"]),
            "num_turns": len(record["turn_evaluations"]),
            "latent_trajectory": [
                item["action"].get("next_latent_id") for item in record["transcript"]
            ],
        }
        for index, record in enumerate(transcripts)
    ]


def latent_transitions(record: Mapping[str, Any]) -> list[tuple[int, int]]:
    previous_by_agent = {"A": 0, "B": 0}
    transitions: list[tuple[int, int]] = []
    for item in record["transcript"]:
        agent_id = str(item["agent_id"])
        current = int(item["action"].get("next_latent_id", 0))
        transitions.append((previous_by_agent.get(agent_id, 0), current))
        previous_by_agent[agent_id] = current
    return transitions


def pareto_efficiency(transcripts: Sequence[Mapping[str, Any]]) -> float:
    if not transcripts:
        return 0.0
    scores = []
    for record in transcripts:
        max_turns = max(1, len(record["turn_evaluations"]))
        if not record["success"]:
            scores.append(0.0)
        else:
            success_turn = next(
                (
                    int(item["turn_index"]) + 1
                    for item in record["turn_evaluations"]
                    if item["exact_match"]
                ),
                max_turns,
            )
            scores.append(1.0 / success_turn)
    return mean_or_zero(scores)


def fairness_score(transcripts: Sequence[Mapping[str, Any]]) -> float:
    lengths_by_agent: dict[str, list[int]] = {"A": [], "B": []}
    for record in transcripts:
        for item in record["transcript"]:
            lengths_by_agent[str(item["agent_id"])].append(
                len(str(item["action"].get("message_text", "")))
            )
    a_mean = mean_or_zero(lengths_by_agent["A"])
    b_mean = mean_or_zero(lengths_by_agent["B"])
    denominator = max(a_mean, b_mean, 1.0)
    return max(0.0, 1.0 - abs(a_mean - b_mean) / denominator)


def classify_emotion(text: str) -> str:
    lower = text.lower()
    keyword_map = {
        "joy": ("嬉", "楽", "喜", "ありがとう", "よい"),
        "sadness": ("悲", "残念", "つら", "難しい"),
        "anger": ("怒", "不満", "だめ"),
        "fear": ("怖", "不安", "心配"),
        "trust": ("協力", "共有", "確認", "統合"),
        "surprise": ("驚", "意外"),
    }
    for label, keywords in keyword_map.items():
        if any(keyword in lower for keyword in keywords):
            return label
    return "neutral"


def mean_or_zero(values: Sequence[float | int]) -> float:
    return statistics.fmean(values) if values else 0.0


def entropy(values: Sequence[Any]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def mutual_information(pairs: Sequence[tuple[Any, Any]]) -> float:
    if not pairs:
        return 0.0
    joint = Counter(pairs)
    left = Counter(item[0] for item in pairs)
    right = Counter(item[1] for item in pairs)
    total = len(pairs)
    value = 0.0
    for (left_value, right_value), count in joint.items():
        p_xy = count / total
        p_x = left[left_value] / total
        p_y = right[right_value] / total
        value += p_xy * math.log(p_xy / (p_x * p_y))
    return value


def write_metrics_csv(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "num_tasks",
        "mean_reward",
        "agreement_rate",
        "pareto_efficiency",
        "fairness",
        "mean_utterance_length",
        "parser_failure_rate",
        "latent_usage_entropy",
        "latent_transition_entropy",
        "task_state_latent_mutual_information",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def write_report(
    summaries: Sequence[Mapping[str, Any]],
    transcripts: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    metrics_csv_path: Path,
    transcript_path: Path,
    heatmap_path: Path,
    reward_curve_path: Path,
    emotion_distribution_path: Path,
    transcript_sample_count: int,
    failure_sample_count: int,
) -> None:
    lines = [
        "# Evaluation Report",
        "",
        "## Outputs",
        "",
        f"- Metrics CSV: `{metrics_csv_path}`",
        f"- Transcripts: `{transcript_path}`",
        f"- Latent transition heatmap: `{heatmap_path}`",
        f"- Reward curve: `{reward_curve_path}`",
        f"- Emotion distribution: `{emotion_distribution_path}`",
        "",
        "## Quantitative Summary",
        "",
        "| Variant | Mean reward | Agreement | Parser failures | Latent transition entropy | MI(task, latent) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| {variant} | {mean_reward:.3f} | {agreement_rate:.3f} | "
            "{parser_failure_rate:.3f} | {latent_transition_entropy:.3f} | "
            "{task_state_latent_mutual_information:.3f} |".format(**summary)
        )

    lines.extend(["", "## Transcript Samples", ""])
    for record in transcripts[:transcript_sample_count]:
        lines.extend(format_transcript_sample(record))

    failures = [record for record in transcripts if not record["success"]]
    if failure_sample_count and failures:
        lines.extend(["", "## Failure Samples", ""])
        for record in failures[:failure_sample_count]:
            lines.extend(format_transcript_sample(record))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_transcript_sample(record: Mapping[str, Any]) -> list[str]:
    lines = [
        f"### {record.get('variant', 'unknown')} episode {record.get('episode_index', 0)}",
        "",
        f"- Answer: `{record['task']['answer']}`",
        f"- Success: `{record['success']}`",
        f"- Total reward: `{record['total_reward']:.3f}`",
        f"- Latent trajectory: `{[item['action'].get('next_latent_id') for item in record['transcript']]}`",
        "",
    ]
    for item in record["transcript"]:
        action = item["action"]
        lines.append(
            f"- Agent {item['agent_id']} t={item['turn_index']} "
            f"latent={action.get('next_latent_id')}: {action.get('message_text', '')}"
        )
    lines.append("")
    return lines


def write_latent_heatmap(transcripts: Sequence[Mapping[str, Any]], path: Path) -> None:
    transitions = Counter(
        transition for record in transcripts for transition in latent_transitions(record)
    )
    ids = sorted({value for pair in transitions for value in pair}) or [0]
    cell = 28
    margin = 70
    width = margin + cell * len(ids) + 20
    height = margin + cell * len(ids) + 20
    max_count = max(transitions.values(), default=1)
    parts = svg_header(width, height)
    parts.append('<text x="10" y="24" font-size="14">Latent transitions</text>')
    for x_index, x_value in enumerate(ids):
        parts.append(
            f'<text x="{margin + x_index * cell + 8}" y="{margin - 10}" font-size="10">{x_value}</text>'
        )
    for y_index, y_value in enumerate(ids):
        parts.append(
            f'<text x="{margin - 24}" y="{margin + y_index * cell + 18}" font-size="10">{y_value}</text>'
        )
        for x_index, x_value in enumerate(ids):
            count = transitions.get((y_value, x_value), 0)
            intensity = int(255 - 200 * (count / max_count))
            parts.append(
                f'<rect x="{margin + x_index * cell}" y="{margin + y_index * cell}" '
                f'width="{cell}" height="{cell}" fill="rgb({intensity},{intensity},255)" '
                'stroke="#333" stroke-width="0.5"/>'
            )
            if count:
                parts.append(
                    f'<text x="{margin + x_index * cell + 8}" y="{margin + y_index * cell + 18}" '
                    f'font-size="10">{count}</text>'
                )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_reward_curve(transcripts: Sequence[Mapping[str, Any]], path: Path) -> None:
    grouped: dict[str, list[float]] = {}
    for record in transcripts:
        grouped.setdefault(str(record.get("variant", "unknown")), []).append(
            float(record["total_reward"])
        )
    width, height = 720, 320
    left, top, plot_width, plot_height = 50, 30, 620, 240
    all_rewards = [reward for rewards in grouped.values() for reward in rewards]
    max_reward = max(all_rewards, default=1.0)
    max_points = max((len(rewards) for rewards in grouped.values()), default=1)
    colors = ["#2f6fbb", "#c44e52", "#55a868", "#8172b3", "#ccb974", "#64b5cd", "#444444"]
    parts = svg_header(width, height)
    parts.append('<text x="10" y="20" font-size="14">Reward curve</text>')
    parts.append(
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="white" stroke="#333"/>'
    )
    for index, (variant, rewards) in enumerate(sorted(grouped.items())):
        points = []
        for step, reward in enumerate(rewards):
            x = left + (plot_width * step / max(1, max_points - 1))
            y = top + plot_height - (plot_height * reward / max(max_reward, 1e-9))
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[index % len(colors)]
        if len(points) == 1:
            x, y = points[0].split(",")
            parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
        else:
            parts.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
        parts.append(
            f'<text x="{left}" y="{top + plot_height + 18 + index * 14}" '
            f'font-size="11" fill="{color}">{variant}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_emotion_distribution(transcripts: Sequence[Mapping[str, Any]], path: Path) -> None:
    counts = Counter(
        classify_emotion(item["action"].get("message_text", ""))
        for record in transcripts
        for item in record["transcript"]
    )
    width, height = 640, 300
    left, top, bar_height = 150, 40, 24
    max_count = max(counts.values(), default=1)
    parts = svg_header(width, height)
    parts.append('<text x="10" y="22" font-size="14">Emotion distribution</text>')
    for index, (label, count) in enumerate(sorted(counts.items())):
        y = top + index * (bar_height + 8)
        bar_width = 420 * count / max_count
        parts.append(f'<text x="10" y="{y + 17}" font-size="12">{label}</text>')
        parts.append(
            f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" fill="#55a868"/>'
        )
        parts.append(f'<text x="{left + bar_width + 6}" y="{y + 17}" font-size="12">{count}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
