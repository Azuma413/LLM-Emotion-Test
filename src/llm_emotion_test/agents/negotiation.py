from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from llm_emotion_test.models.latent import LatentMarkerSpec, parse_latent_id
from llm_emotion_test.tasks.hidden_constraints import (
    Code,
    Constraint,
    HiddenConstraintProblem,
    enumerate_codes,
    filter_codes,
)


@dataclass(frozen=True)
class AgentAction:
    message_text: str
    next_latent_id: int
    proposal: str | None = None
    raw_text: str | None = None
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_text": self.message_text,
            "next_latent_id": self.next_latent_id,
            "proposal": self.proposal,
            "raw_text": self.raw_text,
            "parse_error": self.parse_error,
        }


class NegotiationAgent(Protocol):
    def act(self, observation: Mapping[str, Any]) -> AgentAction:
        ...


class RuleBasedConstraintSharingAgent:
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
        undisclosed = [
            constraint for constraint in private_constraints if constraint.text not in disclosed
        ]
        if undisclosed:
            message = " ".join(constraint.text for constraint in undisclosed)
        else:
            message = "これまで共有した制約を統合して候補を絞りましょう。"
        return AgentAction(
            message_text=message,
            next_latent_id=self.latent_id,
            proposal=observation.get("known_answer"),
            raw_text=message,
        )


class LLMNegotiationAgent:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        agent_id: str,
        num_latents: int,
        marker_template: str,
        generation_max_new_tokens: int = 128,
        fallback: str = "previous",
        neutral_latent_id: int = 0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.agent_id = agent_id
        self.num_latents = num_latents
        self.marker_template = marker_template
        self.generation_max_new_tokens = generation_max_new_tokens
        self.fallback = fallback
        self.neutral_latent_id = neutral_latent_id

    def act(self, observation: Mapping[str, Any]) -> AgentAction:
        import torch

        prompt = build_agent_prompt(observation)
        current_latent_id = int(observation["current_latent_id"])
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        device = next(self.model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_ids=torch.tensor([current_latent_id], device=device, dtype=torch.long),
            max_new_tokens=self.generation_max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        return parse_agent_action(
            generated_text,
            previous_latent_id=current_latent_id,
            num_latents=self.num_latents,
            marker_template=self.marker_template,
            fallback=self.fallback,
            neutral_latent_id=self.neutral_latent_id,
        )


class RuleBasedThirdPartyAnswerer:
    def __init__(self, *, digits: Sequence[int], allow_repeats: bool) -> None:
        self.digits = list(digits)
        self.allow_repeats = allow_repeats

    def answer(
        self,
        *,
        problem: HiddenConstraintProblem,
        transcript: Sequence[Mapping[str, Any]],
    ) -> str | None:
        disclosed_constraints = constraints_disclosed_in_transcript(problem, transcript)
        candidates = filter_codes(
            enumerate_codes(
                code_length=len(problem.code),
                digits=self.digits,
                allow_repeats=self.allow_repeats,
            ),
            disclosed_constraints,
        )
        if len(candidates) == 1:
            return code_to_text(candidates[0])
        return None


def build_agent_prompt(observation: Mapping[str, Any]) -> str:
    lines = [
        "あなたは協調コード推定タスクのエージェントです。",
        f"あなたのID: Agent {observation['agent_id']}",
        "自分の制約:",
    ]
    lines.extend(f"- {constraint.text}" for constraint in observation["private_constraints"])
    lines.append("会話履歴:")
    for item in observation["transcript"]:
        lines.append(f"Agent {item['agent_id']}: {item['action']['message_text']}")
    lines.extend(
        [
            "次の発話では、相手に有用な制約または推定を共有してください。",
            "回答候補がある場合は <answer>1234</answer> の形式で含めてください。",
            "最後に必ず <|emotion|>000<|/emotion|> の形式で latent を出力してください。",
        ]
    )
    return "\n".join(lines)


def parse_agent_action(
    text: str,
    *,
    previous_latent_id: int,
    num_latents: int,
    marker_template: str,
    fallback: str = "previous",
    neutral_latent_id: int = 0,
) -> AgentAction:
    parse_error = None
    marker_spec = LatentMarkerSpec(marker_template)
    marker_matches = list(marker_spec.pattern().finditer(text))
    if not marker_matches:
        parse_error = "Generated text does not contain a latent marker"
    elif not any(0 <= int(match.group("latent_id")) < num_latents for match in marker_matches):
        parse_error = "Generated text does not contain a valid latent marker"
    try:
        latent_id = parse_latent_id(
            text,
            num_latents=num_latents,
            marker_template=marker_template,
            previous_latent_id=previous_latent_id,
            fallback=fallback,
            neutral_latent_id=neutral_latent_id,
        )
    except ValueError as exc:
        latent_id = previous_latent_id
        parse_error = str(exc)

    proposal = parse_answer(text)
    message_text = strip_protocol_markers(text, marker_template=marker_template).strip()
    return AgentAction(
        message_text=message_text,
        next_latent_id=latent_id,
        proposal=proposal,
        raw_text=text,
        parse_error=parse_error,
    )


def parse_answer(text: str) -> str | None:
    match = re.search(r"<answer>\s*(?P<answer>\d{4,5})\s*</answer>", text)
    if match is None:
        return None
    return match.group("answer")


def strip_protocol_markers(text: str, *, marker_template: str) -> str:
    without_answer = re.sub(r"<answer>\s*\d{4,5}\s*</answer>", "", text)
    start = re.escape(marker_template.split("{latent_id", maxsplit=1)[0])
    end = re.escape(marker_template.rsplit("}", maxsplit=1)[-1])
    return re.sub(rf"{start}\s*\d+\s*{end}", "", without_answer)


def constraints_disclosed_in_transcript(
    problem: HiddenConstraintProblem,
    transcript: Sequence[Mapping[str, Any]],
) -> list[Constraint]:
    by_text = {constraint.text: constraint for constraint in problem.all_constraints}
    disclosed: list[Constraint] = []
    for item in transcript:
        message = item["action"]["message_text"]
        for text, constraint in by_text.items():
            if text in message and constraint not in disclosed:
                disclosed.append(constraint)
    return disclosed


def code_to_text(code: Code) -> str:
    return "".join(str(digit) for digit in code)
