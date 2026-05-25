from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

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
        anchor_token: str = "<|latent_pred|>",
        generation_max_new_tokens: int = 128,
        fallback: str = "previous",
        neutral_latent_id: int = 0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.agent_id = agent_id
        self.num_latents = num_latents
        self.marker_template = marker_template
        self.anchor_token = anchor_token
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
        predicted_latent_id = self._predict_next_latent_id(
            prompt + generated_text,
            current_latent_id=current_latent_id,
        )
        raw_text = generated_text + LatentMarkerSpec(self.marker_template).format(
            predicted_latent_id
        )
        return AgentAction(
            message_text=strip_protocol_markers(raw_text, marker_template=self.marker_template).strip(),
            next_latent_id=predicted_latent_id,
            proposal=parse_answer(raw_text),
            raw_text=raw_text,
            parse_error=None,
        )

    @torch.no_grad()
    def _predict_next_latent_id(self, text: str, *, current_latent_id: int) -> int:
        import torch

        if not hasattr(self.model, "predict_latent"):
            return current_latent_id
        encoded = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            anchor_id = self.tokenizer.convert_tokens_to_ids(self.anchor_token)
        else:
            anchor_ids = self.tokenizer(self.anchor_token, add_special_tokens=False)["input_ids"]
            anchor_id = anchor_ids[0]
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([list(encoded) + [int(anchor_id)]], device=device, dtype=torch.long)
        latent_ids = torch.tensor([current_latent_id], device=device, dtype=torch.long)
        latent_positions = torch.tensor([len(encoded)], device=device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        predicted, _distances = self.model.predict_latent(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_ids=latent_ids,
            latent_positions=latent_positions,
        )
        return int(predicted[0].detach().cpu())


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


class LLMThirdPartyAnswerer:
    def __init__(self, config) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from llm_emotion_test.models.loader import resolve_torch_dtype

        tokenizer_id = config.rl_task.third_party_tokenizer or config.rl_task.third_party_model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=config.rl_task.third_party_trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs: dict[str, Any] = {
            "trust_remote_code": config.rl_task.third_party_trust_remote_code,
        }
        if config.rl_task.third_party_device_map is not None:
            kwargs["device_map"] = config.rl_task.third_party_device_map
        dtype = resolve_torch_dtype(config.rl_task.third_party_torch_dtype)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            config.rl_task.third_party_model,
            **kwargs,
        )
        self.config = config

    def answer(
        self,
        *,
        problem: HiddenConstraintProblem,
        transcript: Sequence[Mapping[str, Any]],
    ) -> str | None:
        import torch

        prompt = build_third_party_prompt(problem, transcript)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        do_sample = self.config.rl_task.third_party_temperature > 0.0
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.rl_task.third_party_max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            kwargs["temperature"] = self.config.rl_task.third_party_temperature
            kwargs["top_p"] = self.config.rl_task.third_party_top_p
        with torch.no_grad():
            output_ids = self.model.generate(**encoded, **kwargs)
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        generated = decoded[len(prompt) :] if decoded.startswith(prompt) else decoded
        return parse_answer(generated)


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
    # lines.append("次の発話では、相手に有用な制約を共有してください。")
    return "\n".join(lines)


def build_third_party_prompt(
    problem: HiddenConstraintProblem,
    transcript: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "あなたは協調コード推定タスクの第三者評価器です。",
        "会話履歴だけを使って、現時点で推定できるコードを1つ出力してください。",
        f"コードは{len(problem.code)}桁で、形式は必ず <answer>1234</answer> です。",
        "会話履歴:",
    ]
    for item in transcript:
        lines.append(f"Agent {item['agent_id']}: {item['action']['message_text']}")
    lines.append("暫定回答:")
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
