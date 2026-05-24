from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_emotion_test.agents.negotiation import (
    AgentAction,
    LLMThirdPartyAnswerer,
    RuleBasedThirdPartyAnswerer,
)
from llm_emotion_test.config import ExperimentConfig
from llm_emotion_test.tasks.hidden_constraints import (
    HiddenConstraintProblem,
    generate_hidden_constraint_problem,
)


@dataclass
class NegotiationState:
    problem: HiddenConstraintProblem
    active_agent_id: str = "A"
    turn_index: int = 0
    current_latents: dict[str, int] = field(default_factory=lambda: {"A": 0, "B": 0})
    transcript: list[dict[str, Any]] = field(default_factory=list)
    turn_evaluations: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False


class CooperativeHiddenConstraintsEnv:
    agent_ids = ("A", "B")

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        if config.rl_task.third_party_backend == "llm":
            self.answerer = LLMThirdPartyAnswerer(config)
        else:
            self.answerer = RuleBasedThirdPartyAnswerer(
                digits=config.rl_task.digits,
                allow_repeats=config.rl_task.allow_repeats,
            )
        self.state: NegotiationState | None = None

    def reset(self, seed: int | None = None) -> Mapping[str, Any]:
        problem = generate_hidden_constraint_problem(self.config.rl_task, seed=seed)
        self.state = NegotiationState(problem=problem)
        return self.observe("A")

    def observe(self, agent_id: str) -> dict[str, Any]:
        state = self._require_state()
        if agent_id not in self.agent_ids:
            raise ValueError(f"Unknown agent_id: {agent_id}")
        return {
            "agent_id": agent_id,
            "private_constraints": state.problem.agent_constraints[agent_id],
            "transcript": list(state.transcript),
            "turn_index": state.turn_index,
            "current_latent_id": state.current_latents[agent_id],
            "max_turns": self.config.rl_task.max_turns,
            "known_answer": None,
        }

    def step(self, agent_action: AgentAction | Mapping[str, Any]) -> dict[str, Any]:
        state = self._require_state()
        if state.done:
            raise RuntimeError("Episode is already done")
        action = normalize_action(agent_action)
        agent_id = state.active_agent_id
        state.current_latents[agent_id] = action.next_latent_id
        state.transcript.append(
            {
                "turn_index": state.turn_index,
                "agent_id": agent_id,
                "action": action.to_dict(),
            }
        )

        reward = 0.0
        evaluation = None
        if agent_id == "B":
            evaluation = self.evaluate_turn()
            state.turn_evaluations.append(evaluation)
            reward = float(evaluation["reward"])
            state.turn_index += 1
            state.done = (
                bool(evaluation["exact_match"])
                or state.turn_index >= self.config.rl_task.max_turns
            )
            state.active_agent_id = "A"
        else:
            state.active_agent_id = "B"

        return {
            "observation": None if state.done else self.observe(state.active_agent_id),
            "reward": reward,
            "done": state.done,
            "evaluation": evaluation,
        }

    def evaluate_turn(self) -> dict[str, Any]:
        state = self._require_state()
        answer = self.answerer.answer(problem=state.problem, transcript=state.transcript)
        exact_match = answer == state.problem.answer
        format_scores = [
            action_format_score(item["action"])
            for item in state.transcript[-2:]
        ]
        format_score = sum(format_scores) / len(format_scores) if format_scores else 0.0
        reward = (1.0 if exact_match else 0.0) + 0.1 * format_score
        return {
            "turn_index": state.turn_index,
            "provisional_answer": answer,
            "target_answer": state.problem.answer,
            "exact_match": exact_match,
            "format_score": format_score,
            "reward": reward,
        }

    def compute_reward(self) -> float:
        state = self._require_state()
        if not state.turn_evaluations:
            return 0.0
        return float(state.turn_evaluations[-1]["reward"])

    @property
    def is_done(self) -> bool:
        return self._require_state().done

    def transcript_record(self) -> dict[str, Any]:
        state = self._require_state()
        total_reward = sum(item["reward"] for item in state.turn_evaluations)
        return {
            "task": state.problem.to_dict(),
            "transcript": list(state.transcript),
            "turn_evaluations": list(state.turn_evaluations),
            "total_reward": total_reward,
            "success": any(item["exact_match"] for item in state.turn_evaluations),
        }

    def _require_state(self) -> NegotiationState:
        if self.state is None:
            raise RuntimeError("Environment must be reset before use")
        return self.state


def normalize_action(action: AgentAction | Mapping[str, Any]) -> AgentAction:
    if isinstance(action, AgentAction):
        return action
    return AgentAction(
        message_text=str(action.get("message_text", "")),
        next_latent_id=int(action.get("next_latent_id", 0)),
        proposal=action.get("proposal"),
        raw_text=action.get("raw_text"),
        parse_error=action.get("parse_error"),
    )


def action_format_score(action: Mapping[str, Any]) -> float:
    score = 0.0
    if action.get("message_text"):
        score += 0.4
    if isinstance(action.get("next_latent_id"), int):
        score += 0.4
    if action.get("parse_error") is None:
        score += 0.2
    return score


def write_transcripts(records: list[Mapping[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path
