from __future__ import annotations

import json
import math
import random
import statistics
from copy import deepcopy
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
import torch
from torch.nn import functional as F
from transformers import set_seed

from llm_emotion_test.agents.negotiation import (
    AgentAction,
    RuleBasedConstraintSharingAgent,
    build_agent_prompt,
    parse_agent_action,
)
from llm_emotion_test.config import ExperimentConfig
from llm_emotion_test.models.loader import (
    build_soft_prompt_model,
    load_soft_prompt_model_from_checkpoint,
    save_model_checkpoint,
)
from llm_emotion_test.tasks.negotiation_env import (
    CooperativeHiddenConstraintsEnv,
    action_format_score,
    write_transcripts,
)
from llm_emotion_test.training.sft import write_jsonl


@dataclass(frozen=True)
class RolloutStep:
    prompt: str
    generated_text: str
    logprob: float
    agent_id: str
    latent_id: int
    previous_latent_id: int
    reward: float
    team_reward: float
    local_reward: float
    advantage: float
    group_id: str
    episode_index: int
    rollout_index: int
    turn_index: int
    parser_failed: bool = False
    generated_token_ids: list[int] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)
    reward_components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "logprob": self.logprob,
            "agent_id": self.agent_id,
            "latent_id": self.latent_id,
            "previous_latent_id": self.previous_latent_id,
            "reward": self.reward,
            "team_reward": self.team_reward,
            "local_reward": self.local_reward,
            "advantage": self.advantage,
            "group_id": self.group_id,
            "episode_index": self.episode_index,
            "rollout_index": self.rollout_index,
            "turn_index": self.turn_index,
            "parser_failed": self.parser_failed,
            "generated_token_ids": list(self.generated_token_ids),
            "token_logprobs": list(self.token_logprobs),
            "reward_components": dict(self.reward_components),
        }


@dataclass
class RolloutBuffer:
    steps: list[RolloutStep] = field(default_factory=list)

    def add(self, step: RolloutStep) -> None:
        self.steps.append(step)

    def extend(self, steps: Sequence[RolloutStep]) -> None:
        self.steps.extend(steps)

    def to_records(self) -> list[dict[str, Any]]:
        return [step.to_dict() for step in self.steps]

    def latent_usage_entropy(self) -> float:
        counts = Counter(step.latent_id for step in self.steps)
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return -sum((count / total) * math.log(count / total) for count in counts.values())

    def parser_failure_rate(self) -> float:
        if not self.steps:
            return 0.0
        return sum(step.parser_failed for step in self.steps) / len(self.steps)

    def latent_usage_distribution(self) -> dict[str, int]:
        counts = Counter(step.latent_id for step in self.steps)
        return {str(key): counts[key] for key in sorted(counts)}

    def latent_transition_entropy(self) -> float:
        counts = Counter((step.previous_latent_id, step.latent_id) for step in self.steps)
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return -sum((count / total) * math.log(count / total) for count in counts.values())


class TabularLatentPolicy:
    """Small CPU policy used to smoke-test GRPO mechanics before model RL."""

    def __init__(
        self,
        *,
        agent_ids: Sequence[str],
        max_turns: int,
        num_latents: int,
        rng: random.Random,
        logits: Mapping[str, Sequence[Sequence[float]]] | None = None,
    ) -> None:
        self.agent_ids = tuple(agent_ids)
        self.max_turns = max_turns
        self.num_latents = num_latents
        self.rng = rng
        if logits is None:
            self.logits = {
                agent_id: [[0.0] * num_latents for _ in range(max_turns)]
                for agent_id in self.agent_ids
            }
        else:
            self.logits = {
                agent_id: [list(row) for row in rows]
                for agent_id, rows in logits.items()
            }

    def sample(self, *, agent_id: str, turn_index: int) -> tuple[int, float]:
        probabilities = softmax(self.logits[agent_id][turn_index])
        threshold = self.rng.random()
        cumulative = 0.0
        latent_id = len(probabilities) - 1
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if threshold <= cumulative:
                latent_id = index
                break
        return latent_id, math.log(max(probabilities[latent_id], 1e-12))

    def update(self, steps: Sequence[RolloutStep], *, learning_rate: float) -> None:
        for step in steps:
            row = self.logits[step.agent_id][step.turn_index]
            probabilities = softmax(row)
            for latent_id in range(self.num_latents):
                indicator = 1.0 if latent_id == step.latent_id else 0.0
                row[latent_id] += learning_rate * step.advantage * (
                    indicator - probabilities[latent_id]
                )

    def to_dict(self) -> dict[str, Any]:
        return {"logits": self.logits}


def run_rule_based_rl_smoke(config: ExperimentConfig) -> dict[str, Any]:
    if config.rl_task.policy_backend == "llm":
        return train_at_grpo_llm(config)
    return train_at_grpo_smoke(config)


def train_at_grpo_llm(config: ExperimentConfig) -> dict[str, Any]:
    if config.runtime.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("LLM AT-GRPO requires a CUDA GPU unless runtime.device='cpu'")
    set_seed(config.runtime.seed)
    config.output.run_dir.mkdir(parents=True, exist_ok=True)
    config.output.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config.output.config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    model, tokenizer = load_or_create_rl_model(config)
    configure_rl_trainable_parameters(model, config)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    baseline_metrics = evaluate_llm_policy(config, model, tokenizer, seed_offset=10_000)
    fixed_baseline_metrics = evaluate_fixed_baseline(config, seed_offset=30_000)
    all_transcripts: list[dict[str, Any]] = []
    buffer = RolloutBuffer()
    losses: list[float] = []

    start_episode = 0
    for episode_index in range(start_episode, config.rl_task.num_episodes):
        if episode_index >= config.training.max_steps:
            break
        rollout = run_llm_tree_rollout(
            config,
            model,
            tokenizer,
            episode_index=episode_index,
            problem_seed=config.runtime.seed + episode_index,
        )
        steps = rollout["steps"]
        loss = update_llm_policy(
            config,
            model,
            tokenizer,
            optimizer,
            steps,
        )
        losses.append(loss)
        buffer.extend(steps)
        all_transcripts.append(rollout["transcript"])
        save_model_checkpoint(
            model,
            tokenizer,
            config,
            config.output.checkpoints_dir / f"episode-{episode_index + 1:06d}",
        )

    final_checkpoint = save_model_checkpoint(
        model,
        tokenizer,
        config,
        config.output.checkpoints_dir / "final",
    )
    transcript_path = config.output.run_dir / config.rl_task.transcript_filename
    write_transcripts(all_transcripts, transcript_path)
    rollout_buffer_path = config.output.run_dir / "rollout_buffer.jsonl"
    write_jsonl(buffer.to_records(), rollout_buffer_path)
    post_metrics = evaluate_llm_policy(config, model, tokenizer, seed_offset=20_000)

    rewards = [float(record["total_reward"]) for record in all_transcripts]
    successes = [bool(record["success"]) for record in all_transcripts]
    metrics = {
        "mode": "at_grpo_llm",
        "algorithm": "AT-GRPO LLM token-level policy update",
        "num_episodes": len(all_transcripts),
        "rollouts_per_problem": config.rl_task.rollouts_per_problem,
        "num_rollouts": len(all_transcripts),
        "num_rollout_steps": len(buffer.steps),
        "num_branch_groups": len({step.group_id for step in buffer.steps}),
        "num_branch_samples": len(buffer.steps),
        "sampling_scheme": "tree-structured per-agent-turn branching",
        "reward_team_weight": config.rl_task.reward_team_weight,
        "success_rate": sum(successes) / len(successes) if successes else 0.0,
        "mean_total_reward": statistics.fmean(rewards) if rewards else 0.0,
        "mean_policy_loss": statistics.fmean(losses) if losses else 0.0,
        "pre_success_rate": baseline_metrics["success_rate"],
        "pre_mean_total_reward": baseline_metrics["mean_total_reward"],
        "fixed_baseline_success_rate": fixed_baseline_metrics["success_rate"],
        "fixed_baseline_mean_total_reward": fixed_baseline_metrics["mean_total_reward"],
        "post_success_rate": post_metrics["success_rate"],
        "post_mean_total_reward": post_metrics["mean_total_reward"],
        "agreement_rate": post_metrics["success_rate"],
        "reward_trend": rewards,
        "latent_usage_distribution": buffer.latent_usage_distribution(),
        "latent_usage_entropy": buffer.latent_usage_entropy(),
        "latent_transition_entropy": buffer.latent_transition_entropy(),
        "parser_failure_rate": buffer.parser_failure_rate(),
        "transcript_path": str(transcript_path),
        "rollout_buffer_path": str(rollout_buffer_path),
        "checkpoint_path": str(final_checkpoint),
    }
    write_jsonl([metrics], config.output.metrics_path)
    return metrics


def train_at_grpo_smoke(config: ExperimentConfig) -> dict[str, Any]:
    config.output.run_dir.mkdir(parents=True, exist_ok=True)
    config.output.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config.output.config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    rng = random.Random(config.runtime.seed)
    policy, start_episode = load_or_create_policy(config, rng=rng)
    pre_metrics = evaluate_policy(config, policy, seed_offset=10_000)
    fixed_baseline_metrics = evaluate_fixed_baseline(config, seed_offset=30_000)

    all_transcripts: list[dict[str, Any]] = []
    buffer = RolloutBuffer()
    for episode_index in range(start_episode, config.rl_task.num_episodes):
        problem_seed = config.runtime.seed + episode_index
        rollout = run_tree_rollout(
            config,
            policy,
            episode_index=episode_index,
            problem_seed=problem_seed,
        )
        group_steps = rollout["steps"]
        policy.update(
            group_steps,
            learning_rate=config.rl_task.latent_policy_learning_rate,
        )
        buffer.extend(group_steps)
        all_transcripts.append(rollout["transcript"])
        save_rl_checkpoint(config, policy, episode_index=episode_index + 1)

    transcript_path = config.output.run_dir / config.rl_task.transcript_filename
    write_transcripts(all_transcripts, transcript_path)

    rollout_buffer_path = config.output.run_dir / "rollout_buffer.jsonl"
    write_jsonl(buffer.to_records(), rollout_buffer_path)

    post_metrics = evaluate_policy(config, policy, seed_offset=20_000)
    rewards = [float(record["total_reward"]) for record in all_transcripts]
    successes = [bool(record["success"]) for record in all_transcripts]
    reward_trend = rewards
    metrics = {
        "mode": "at_grpo_smoke",
        "algorithm": "AT-GRPO turn-wise grouped advantage",
        "num_episodes": config.rl_task.num_episodes,
        "rollouts_per_problem": config.rl_task.rollouts_per_problem,
        "num_rollouts": len(all_transcripts),
        "num_rollout_steps": len(buffer.steps),
        "num_branch_groups": len({step.group_id for step in buffer.steps}),
        "num_branch_samples": len(buffer.steps),
        "sampling_scheme": "tree-structured per-agent-turn branching",
        "reward_team_weight": config.rl_task.reward_team_weight,
        "success_rate": sum(successes) / len(successes) if successes else 0.0,
        "mean_total_reward": statistics.fmean(rewards) if rewards else 0.0,
        "pre_success_rate": pre_metrics["success_rate"],
        "pre_mean_total_reward": pre_metrics["mean_total_reward"],
        "fixed_baseline_success_rate": fixed_baseline_metrics["success_rate"],
        "fixed_baseline_mean_total_reward": fixed_baseline_metrics["mean_total_reward"],
        "post_success_rate": post_metrics["success_rate"],
        "post_mean_total_reward": post_metrics["mean_total_reward"],
        "agreement_rate": post_metrics["success_rate"],
        "reward_trend": reward_trend,
        "latent_usage_distribution": buffer.latent_usage_distribution(),
        "latent_usage_entropy": buffer.latent_usage_entropy(),
        "latent_transition_entropy": buffer.latent_transition_entropy(),
        "parser_failure_rate": buffer.parser_failure_rate(),
        "transcript_path": str(transcript_path),
        "rollout_buffer_path": str(rollout_buffer_path),
        "checkpoint_path": str(config.output.checkpoints_dir / config.rl_task.checkpoint_filename),
    }
    write_jsonl([metrics], config.output.metrics_path)
    return metrics


def load_or_create_rl_model(config: ExperimentConfig):
    if config.rl_task.resume_from_checkpoint is not None:
        checkpoint_path = config.rl_task.resume_from_checkpoint
        if checkpoint_path.exists() and checkpoint_path.is_dir():
            model, tokenizer = load_soft_prompt_model_from_checkpoint(checkpoint_path, config)
            model.soft_prompt.to(model_device(model))
            return model, tokenizer
    model, tokenizer = build_soft_prompt_model(config)
    model.soft_prompt.to(model_device(model))
    return model, tokenizer


def configure_rl_trainable_parameters(model, config: ExperimentConfig) -> None:
    if not config.rl_task.train_base_model:
        for parameter in model.base_model.parameters():
            parameter.requires_grad = False
    if config.model.use_lora or config.model.use_qlora:
        for name, parameter in model.base_model.named_parameters():
            if "lora_" in name or "modules_to_save" in name:
                parameter.requires_grad = True
    for parameter in model.soft_prompt.parameters():
        parameter.requires_grad = True
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable RL parameters are enabled")


def run_llm_tree_rollout(
    config: ExperimentConfig,
    model,
    tokenizer,
    *,
    episode_index: int,
    problem_seed: int,
) -> dict[str, Any]:
    env = CooperativeHiddenConstraintsEnv(config)
    env.reset(seed=problem_seed)
    completed_steps: list[RolloutStep] = []

    while not env.is_done:
        state = env._require_state()
        agent_id = state.active_agent_id
        observation = env.observe(agent_id)
        group_id = (
            f"env-{episode_index}:agent-{agent_id}:turn-{observation['turn_index']}"
        )
        candidates = sample_llm_candidate_group(
            config,
            env,
            model,
            tokenizer,
            observation=observation,
            episode_index=episode_index,
            group_id=group_id,
        )
        assign_group_advantages(
            candidates,
            epsilon=config.rl_task.grpo_advantage_epsilon,
        )
        best_step = max(candidates, key=lambda step: step.reward)
        env.step(llm_step_to_action(best_step, config))
        completed_steps.extend(candidates)

    transcript = env.transcript_record()
    transcript["episode_index"] = episode_index
    transcript["sampling_scheme"] = "tree-structured per-agent-turn branching"
    return {"transcript": transcript, "steps": completed_steps}


def sample_llm_candidate_group(
    config: ExperimentConfig,
    env: CooperativeHiddenConstraintsEnv,
    model,
    tokenizer,
    *,
    observation: Mapping[str, Any],
    episode_index: int,
    group_id: str,
) -> list[RolloutStep]:
    prompt = build_agent_prompt(observation)
    prompt_token_ids = encode_prompt_ids(
        tokenizer,
        prompt,
        max_prompt_length=max(
            1,
            config.training.max_seq_length - config.training.generation_max_new_tokens,
        ),
    )
    candidates: list[RolloutStep] = []
    for branch_index in range(config.rl_task.rollouts_per_problem):
        generated_token_ids, generated_text, sample_logprob = generate_llm_action_tokens(
            config,
            model,
            tokenizer,
            prompt_token_ids,
            latent_id=int(observation["current_latent_id"]),
        )
        action = parse_agent_action(
            generated_text,
            previous_latent_id=int(observation["current_latent_id"]),
            num_latents=config.soft_prompt.num_latents,
            marker_template=config.soft_prompt.latent_marker_template,
            fallback=config.soft_prompt.invalid_latent_fallback,
            neutral_latent_id=config.soft_prompt.neutral_latent_id,
        )
        team_reward, local_reward, reward_components = score_candidate_action(
            config,
            env,
            action,
        )
        candidates.append(
            RolloutStep(
                prompt=prompt,
                generated_text=generated_text,
                logprob=sample_logprob,
                agent_id=str(observation["agent_id"]),
                latent_id=action.next_latent_id,
                previous_latent_id=int(observation["current_latent_id"]),
                reward=config.rl_task.reward_team_weight * team_reward + local_reward,
                team_reward=team_reward,
                local_reward=local_reward,
                advantage=0.0,
                group_id=group_id,
                episode_index=episode_index,
                rollout_index=branch_index,
                turn_index=int(observation["turn_index"]),
                parser_failed=action.parse_error is not None,
                generated_token_ids=generated_token_ids,
                token_logprobs=[sample_logprob],
                reward_components=reward_components,
            )
        )
    return candidates


@torch.no_grad()
def generate_llm_action_tokens(
    config: ExperimentConfig,
    model,
    tokenizer,
    prompt_token_ids: Sequence[int],
    *,
    latent_id: int,
) -> tuple[list[int], str, float]:
    model.eval()
    device = model_device(model)
    input_ids = torch.tensor([list(prompt_token_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    latent_ids = torch.tensor([latent_id], dtype=torch.long, device=device)
    do_sample = config.rl_task.sampling_temperature > 0.0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": config.training.generation_max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = config.rl_task.sampling_temperature
        generation_kwargs["top_p"] = config.rl_task.sampling_top_p
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        latent_ids=latent_ids,
        **generation_kwargs,
    )
    generated_ids = extract_generated_token_ids(output_ids[0], input_ids[0])
    if not generated_ids:
        generated_ids = [tokenizer.eos_token_id or tokenizer.pad_token_id or 0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    logprob = float(
        sequence_logprob(
            model,
            prompt_token_ids,
            generated_ids,
            latent_id=latent_id,
        )
        .detach()
        .cpu()
    )
    return generated_ids, generated_text, logprob


def update_llm_policy(
    config: ExperimentConfig,
    model,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    steps: Sequence[RolloutStep],
) -> float:
    model.train()
    losses: list[torch.Tensor] = []
    for step in steps:
        if not step.generated_token_ids:
            continue
        prompt_ids = encode_prompt_ids(
            tokenizer,
            step.prompt,
            max_prompt_length=max(
                1,
                config.training.max_seq_length - len(step.generated_token_ids),
            ),
        )
        logprob = sequence_logprob(
            model,
            prompt_ids,
            step.generated_token_ids,
            latent_id=step.previous_latent_id,
        )
        length = max(1, len(step.generated_token_ids))
        advantage = torch.tensor(
            float(step.advantage),
            dtype=logprob.dtype,
            device=logprob.device,
        )
        losses.append(-advantage.detach() * logprob / length)
    if not losses:
        return 0.0
    loss = torch.stack(losses).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        max_norm=1.0,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().cpu())


def sequence_logprob(
    model,
    prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    *,
    latent_id: int,
) -> torch.Tensor:
    device = model_device(model)
    full_ids = list(prompt_token_ids) + list(generated_token_ids)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    latent_ids = torch.tensor([latent_id], dtype=torch.long, device=device)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        latent_ids=latent_ids,
    )
    logits = output.logits[0]
    soft_prompt_length = model.soft_prompt.prompt_length
    prompt_length = len(prompt_token_ids)
    token_logprobs: list[torch.Tensor] = []
    for offset, token_id in enumerate(generated_token_ids):
        token_position = prompt_length + offset
        logits_position = soft_prompt_length + token_position - 1
        if logits_position < 0 or logits_position >= logits.shape[0]:
            continue
        token_logprobs.append(
            F.log_softmax(logits[logits_position], dim=-1)[int(token_id)]
        )
    if not token_logprobs:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    return torch.stack(token_logprobs).sum()


def encode_prompt_ids(tokenizer, prompt: str, *, max_prompt_length: int) -> list[int]:
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(token_ids) > max_prompt_length:
        token_ids = token_ids[-max_prompt_length:]
    if not token_ids:
        token_ids = [tokenizer.eos_token_id or tokenizer.pad_token_id or 0]
    return [int(token_id) for token_id in token_ids]


def extract_generated_token_ids(
    output_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
) -> list[int]:
    output = output_ids.detach().cpu().tolist()
    prompt = prompt_ids.detach().cpu().tolist()
    if len(output) > len(prompt) and output[: len(prompt)] == prompt:
        return [int(token_id) for token_id in output[len(prompt) :]]
    return [int(token_id) for token_id in output]


def model_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def llm_step_to_action(step: RolloutStep, config: ExperimentConfig) -> AgentAction:
    return parse_agent_action(
        step.generated_text,
        previous_latent_id=step.previous_latent_id,
        num_latents=config.soft_prompt.num_latents,
        marker_template=config.soft_prompt.latent_marker_template,
        fallback=config.soft_prompt.invalid_latent_fallback,
        neutral_latent_id=config.soft_prompt.neutral_latent_id,
    )


def evaluate_llm_policy(
    config: ExperimentConfig,
    model,
    tokenizer,
    *,
    seed_offset: int,
) -> dict[str, float]:
    transcripts = [
        run_llm_tree_rollout(
            config,
            model,
            tokenizer,
            episode_index=index,
            problem_seed=config.runtime.seed + seed_offset + index,
        )["transcript"]
        for index in range(config.rl_task.eval_episodes)
    ]
    rewards = [float(record["total_reward"]) for record in transcripts]
    successes = [bool(record["success"]) for record in transcripts]
    return {
        "success_rate": sum(successes) / len(successes) if successes else 0.0,
        "mean_total_reward": statistics.fmean(rewards) if rewards else 0.0,
    }


def run_policy_rollout(
    config: ExperimentConfig,
    policy: TabularLatentPolicy,
    *,
    episode_index: int,
    rollout_index: int,
    rollout_group_id: str,
    problem_seed: int,
) -> dict[str, Any]:
    env = CooperativeHiddenConstraintsEnv(config)
    env.reset(seed=problem_seed)
    speaker = RuleBasedConstraintSharingAgent(latent_id=0)
    completed_steps: list[RolloutStep] = []

    while not env.is_done:
        state = env._require_state()
        agent_id = state.active_agent_id
        observation = env.observe(agent_id)
        group_id = (
            f"{rollout_group_id}:agent-{agent_id}:turn-{observation['turn_index']}"
        )
        latent_id, logprob = policy.sample(
            agent_id=str(observation["agent_id"]),
            turn_index=int(observation["turn_index"]),
        )
        action = build_smoke_candidate_action(
            speaker,
            observation,
            latent_id=latent_id,
        )
        result = env.step(action)
        team_reward, local_reward, reward_components = compute_action_rewards(
            config,
            action,
            result.get("evaluation"),
        )
        completed_steps.append(
            RolloutStep(
                prompt=observation_to_prompt(observation),
                generated_text=action.raw_text or action.message_text,
                logprob=logprob,
                agent_id=str(observation["agent_id"]),
                latent_id=latent_id,
                previous_latent_id=int(observation["current_latent_id"]),
                reward=config.rl_task.reward_team_weight * team_reward + local_reward,
                team_reward=team_reward,
                local_reward=local_reward,
                advantage=0.0,
                group_id=group_id,
                episode_index=episode_index,
                rollout_index=rollout_index,
                turn_index=int(observation["turn_index"]),
                parser_failed=action.parse_error is not None,
                generated_token_ids=[],
                token_logprobs=[logprob],
                reward_components=reward_components,
            )
        )

    transcript = env.transcript_record()
    transcript["episode_index"] = episode_index
    transcript["rollout_index"] = rollout_index
    transcript["group_id"] = rollout_group_id
    return {"transcript": transcript, "steps": completed_steps}


def run_tree_rollout(
    config: ExperimentConfig,
    policy: TabularLatentPolicy,
    *,
    episode_index: int,
    problem_seed: int,
) -> dict[str, Any]:
    env = CooperativeHiddenConstraintsEnv(config)
    env.reset(seed=problem_seed)
    speaker = RuleBasedConstraintSharingAgent(latent_id=0)
    completed_steps: list[RolloutStep] = []

    while not env.is_done:
        state = env._require_state()
        agent_id = state.active_agent_id
        observation = env.observe(agent_id)
        group_id = (
            f"env-{episode_index}:agent-{agent_id}:turn-{observation['turn_index']}"
        )
        candidates = sample_candidate_group(
            config,
            env,
            speaker,
            policy,
            observation=observation,
            episode_index=episode_index,
            group_id=group_id,
        )
        assign_group_advantages(
            candidates,
            epsilon=config.rl_task.grpo_advantage_epsilon,
        )
        best_step = max(candidates, key=lambda step: step.reward)
        env.step(
            AgentAction(
                message_text=best_step.generated_text,
                next_latent_id=best_step.latent_id,
                raw_text=best_step.generated_text,
                parse_error=None if not best_step.parser_failed else "candidate parse failed",
            )
        )
        completed_steps.extend(candidates)

    transcript = env.transcript_record()
    transcript["episode_index"] = episode_index
    transcript["sampling_scheme"] = "tree-structured per-agent-turn branching"
    return {"transcript": transcript, "steps": completed_steps}


def run_rollout(
    config: ExperimentConfig,
    policy: TabularLatentPolicy,
    *,
    episode_index: int,
    rollout_index: int,
    group_id: str,
    problem_seed: int,
) -> dict[str, Any]:
    rollout = run_policy_rollout(
        config,
        policy,
        episode_index=episode_index,
        rollout_index=rollout_index,
        rollout_group_id=group_id,
        problem_seed=problem_seed,
    )
    return rollout


def sample_candidate_group(
    config: ExperimentConfig,
    env: CooperativeHiddenConstraintsEnv,
    speaker: RuleBasedConstraintSharingAgent,
    policy: TabularLatentPolicy,
    *,
    observation: Mapping[str, Any],
    episode_index: int,
    group_id: str,
) -> list[RolloutStep]:
    candidates: list[RolloutStep] = []
    for branch_index in range(config.rl_task.rollouts_per_problem):
        latent_id, logprob = policy.sample(
            agent_id=str(observation["agent_id"]),
            turn_index=int(observation["turn_index"]),
        )
        action = build_smoke_candidate_action(
            speaker,
            observation,
            latent_id=latent_id,
        )
        team_reward, local_reward, reward_components = score_candidate_action(
            config, env, action
        )
        candidates.append(
            RolloutStep(
                prompt=observation_to_prompt(observation),
                generated_text=action.raw_text or action.message_text,
                logprob=logprob,
                agent_id=str(observation["agent_id"]),
                latent_id=latent_id,
                previous_latent_id=int(observation["current_latent_id"]),
                reward=config.rl_task.reward_team_weight * team_reward + local_reward,
                team_reward=team_reward,
                local_reward=local_reward,
                advantage=0.0,
                group_id=group_id,
                episode_index=episode_index,
                rollout_index=branch_index,
                turn_index=int(observation["turn_index"]),
                parser_failed=action.parse_error is not None,
                generated_token_ids=[],
                token_logprobs=[logprob],
                reward_components=reward_components,
            )
        )
    return candidates


def compute_action_rewards(
    config: ExperimentConfig,
    action: AgentAction,
    evaluation: Mapping[str, Any] | None,
) -> tuple[float, float, dict[str, float]]:
    local_reward = action_format_score(action.to_dict())
    team_reward = 0.0
    if evaluation is not None:
        team_reward = 1.0 if evaluation["exact_match"] else 0.0
    components = {
        "team_exact_match": team_reward,
        "action_format": local_reward,
        "parser_failure_penalty": -1.0 if action.parse_error is not None else 0.0,
    }
    return team_reward, local_reward, components


def score_candidate_action(
    config: ExperimentConfig,
    env: CooperativeHiddenConstraintsEnv,
    action: AgentAction,
) -> tuple[float, float, dict[str, float]]:
    trial_env = deepcopy(env)
    result = trial_env.step(action)
    team_reward, local_reward, reward_components = compute_action_rewards(
        config,
        action,
        result.get("evaluation"),
    )
    return team_reward, local_reward, reward_components


def build_smoke_candidate_action(
    speaker: RuleBasedConstraintSharingAgent,
    observation: Mapping[str, Any],
    *,
    latent_id: int,
) -> AgentAction:
    base_action = speaker.act(observation)
    mode = latent_id % 4
    if mode == 0:
        return AgentAction(
            message_text=base_action.message_text,
            next_latent_id=latent_id,
            proposal=base_action.proposal,
            raw_text=base_action.raw_text,
            parse_error=base_action.parse_error,
        )
    if mode == 1:
        first_constraint = str(base_action.message_text).split("。", maxsplit=1)[0]
        message = f"{first_constraint}。" if first_constraint else ""
        return AgentAction(
            message_text=message,
            next_latent_id=latent_id,
            proposal=base_action.proposal,
            raw_text=message,
            parse_error=None,
        )
    if mode == 2:
        return AgentAction(
            message_text="",
            next_latent_id=latent_id,
            proposal=None,
            raw_text="",
            parse_error=None,
        )
    return AgentAction(
        message_text=base_action.message_text,
        next_latent_id=latent_id,
        proposal=base_action.proposal,
        raw_text=base_action.raw_text,
        parse_error="synthetic malformed protocol branch",
    )


def assign_at_grpo_advantages(
    rollouts: Sequence[Mapping[str, Any]],
    *,
    epsilon: float,
) -> None:
    by_turn: dict[tuple[int, str, int], list[RolloutStep]] = defaultdict(list)
    for rollout in rollouts:
        for step in rollout["steps"]:
            by_turn[(step.episode_index, step.agent_id, step.turn_index)].append(step)

    replacements: dict[int, float] = {}
    for steps in by_turn.values():
        rewards = [step.reward for step in steps]
        mean_reward = statistics.fmean(rewards)
        std_reward = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        scale = std_reward + epsilon
        for step in steps:
            replacements[id(step)] = (step.reward - mean_reward) / scale

    for rollout in rollouts:
        rollout["steps"][:] = [
            RolloutStep(**{**step.to_dict(), "advantage": replacements[id(step)]})
            for step in rollout["steps"]
        ]


def assign_group_advantages(steps: list[RolloutStep], *, epsilon: float) -> None:
    rewards = [step.reward for step in steps]
    mean_reward = statistics.fmean(rewards)
    std_reward = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    scale = std_reward + epsilon
    for index, step in enumerate(list(steps)):
        steps[index] = RolloutStep(
            **{**step.to_dict(), "advantage": (step.reward - mean_reward) / scale}
        )


def evaluate_policy(
    config: ExperimentConfig,
    policy: TabularLatentPolicy,
    *,
    seed_offset: int,
) -> dict[str, float]:
    transcripts = [
        run_rollout(
            config,
            policy,
            episode_index=index,
            rollout_index=0,
            group_id=f"eval-{index}",
            problem_seed=config.runtime.seed + seed_offset + index,
        )["transcript"]
        for index in range(config.rl_task.eval_episodes)
    ]
    rewards = [float(record["total_reward"]) for record in transcripts]
    successes = [bool(record["success"]) for record in transcripts]
    return {
        "success_rate": sum(successes) / len(successes) if successes else 0.0,
        "mean_total_reward": statistics.fmean(rewards) if rewards else 0.0,
    }


def evaluate_fixed_baseline(
    config: ExperimentConfig,
    *,
    seed_offset: int,
) -> dict[str, float]:
    transcripts = [
        run_fixed_agent_episode(
            config,
            episode_index=index,
            problem_seed=config.runtime.seed + seed_offset + index,
        )
        for index in range(config.rl_task.eval_episodes)
    ]
    rewards = [float(record["total_reward"]) for record in transcripts]
    successes = [bool(record["success"]) for record in transcripts]
    return {
        "success_rate": sum(successes) / len(successes) if successes else 0.0,
        "mean_total_reward": statistics.fmean(rewards) if rewards else 0.0,
    }


def run_fixed_agent_episode(
    config: ExperimentConfig,
    *,
    episode_index: int,
    problem_seed: int,
) -> dict[str, Any]:
    env = CooperativeHiddenConstraintsEnv(config)
    env.reset(seed=problem_seed)
    agents = {
        agent_id: RuleBasedConstraintSharingAgent(latent_id=0)
        for agent_id in CooperativeHiddenConstraintsEnv.agent_ids
    }
    while not env.is_done:
        agent_id = env._require_state().active_agent_id
        env.step(agents[agent_id].act(env.observe(agent_id)))
    transcript = env.transcript_record()
    transcript["episode_index"] = episode_index
    transcript["baseline"] = "rule_based_constraint_sharing"
    return transcript


def load_or_create_policy(
    config: ExperimentConfig,
    *,
    rng: random.Random,
) -> tuple[TabularLatentPolicy, int]:
    checkpoint_path = config.rl_task.resume_from_checkpoint
    if checkpoint_path is not None and checkpoint_path.exists():
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        return (
            TabularLatentPolicy(
                agent_ids=CooperativeHiddenConstraintsEnv.agent_ids,
                max_turns=config.rl_task.max_turns,
                num_latents=config.soft_prompt.num_latents,
                rng=rng,
                logits=data["policy"]["logits"],
            ),
            int(data.get("episode_index", 0)),
        )
    return (
        TabularLatentPolicy(
            agent_ids=CooperativeHiddenConstraintsEnv.agent_ids,
            max_turns=config.rl_task.max_turns,
            num_latents=config.soft_prompt.num_latents,
            rng=rng,
        ),
        0,
    )


def save_rl_checkpoint(
    config: ExperimentConfig,
    policy: TabularLatentPolicy,
    *,
    episode_index: int,
) -> Path:
    checkpoint_path = config.output.checkpoints_dir / config.rl_task.checkpoint_filename
    checkpoint = {
        "episode_index": episode_index,
        "policy": policy.to_dict(),
        "config": {
            "num_latents": config.soft_prompt.num_latents,
            "max_turns": config.rl_task.max_turns,
        },
    }
    checkpoint_path.write_text(
        json.dumps(checkpoint, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return checkpoint_path


def observation_to_prompt(observation: Mapping[str, Any]) -> str:
    constraints = "\n".join(
        f"- {constraint.text}" for constraint in observation["private_constraints"]
    )
    history = "\n".join(
        f"Agent {item['agent_id']}: {item['action']['message_text']}"
        for item in observation["transcript"]
    )
    return (
        f"Agent {observation['agent_id']}\n"
        f"turn={observation['turn_index']}\n"
        f"latent={observation['current_latent_id']}\n"
        f"constraints:\n{constraints}\n"
        f"history:\n{history}"
    )


def softmax(logits: Sequence[float]) -> list[float]:
    max_logit = max(logits)
    exp_values = [math.exp(value - max_logit) for value in logits]
    total = sum(exp_values)
    return [value / total for value in exp_values]
