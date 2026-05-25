from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from llm_emotion_test.models.latent import LatentMarkerSpec, parse_latent_id
from llm_emotion_test.models.soft_prompt import (
    SoftPromptCausalLM,
    SoftPromptEmbedding,
    load_soft_prompt,
)


class DummyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        self.last_inputs_embeds_shape: tuple[int, int, int] | None = None

    def get_input_embeddings(self):
        return self.embedding

    def resize_token_embeddings(self, size: int):
        self.embedding = nn.Embedding(size, self.config.hidden_size)
        return self.embedding

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask=None,
        labels=None,
        output_hidden_states=False,
        **_kwargs,
    ):
        self.last_inputs_embeds_shape = tuple(inputs_embeds.shape)
        logits = self.head(inputs_embeds)
        loss = None
        if labels is not None:
            loss = logits.sum() * 0
        hidden_states = (inputs_embeds,) if output_hidden_states else None
        return SimpleNamespace(
            logits=logits,
            loss=loss,
            attention_mask=attention_mask,
            hidden_states=hidden_states,
        )

    def generate(self, *, inputs_embeds, attention_mask=None, **_kwargs):
        self.last_inputs_embeds_shape = tuple(inputs_embeds.shape)
        return torch.zeros(inputs_embeds.shape[0], 2, dtype=torch.long)

    def save_pretrained(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "dummy.txt").write_text("saved", encoding="utf-8")


class ModelOutputDummyCausalLM(DummyCausalLM):
    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask=None,
        labels=None,
        output_hidden_states=False,
        **_kwargs,
    ):
        self.last_inputs_embeds_shape = tuple(inputs_embeds.shape)
        logits = self.head(inputs_embeds)
        loss = logits.sum() * 0 if labels is not None else None
        hidden_states = (inputs_embeds,) if output_hidden_states else None
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )


def test_soft_prompt_forward_prepends_prompt_embeddings() -> None:
    base_model = DummyCausalLM()
    soft_prompt = SoftPromptEmbedding(
        num_latents=3,
        prompt_length=4,
        hidden_size=8,
        init_strategy="zeros",
    )
    model = SoftPromptCausalLM(base_model, soft_prompt)

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    latent_ids = torch.tensor([0, 2])
    labels = input_ids.clone()
    output = model(input_ids=input_ids, latent_ids=latent_ids, labels=labels)

    assert base_model.last_inputs_embeds_shape == (2, 7, 8)
    assert output.logits.shape == (2, 7, 16)
    assert output.loss is not None


def test_soft_prompt_forward_returns_latent_loss() -> None:
    base_model = DummyCausalLM()
    soft_prompt = SoftPromptEmbedding(
        num_latents=3,
        prompt_length=4,
        hidden_size=8,
        init_strategy="zeros",
    )
    model = SoftPromptCausalLM(base_model, soft_prompt, latent_loss_weight=0.5)

    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, 2, 3, -100]])
    output = model(
        input_ids=input_ids,
        latent_ids=torch.tensor([0]),
        labels=labels,
        target_latent_ids=torch.tensor([2]),
        latent_positions=torch.tensor([3]),
    )

    assert output.text_loss is not None
    assert output.latent_loss is not None
    assert output.pred_latent.shape == (1, 8)
    assert output.loss is not None


def test_soft_prompt_forward_preserves_model_output_indexing() -> None:
    base_model = ModelOutputDummyCausalLM()
    soft_prompt = SoftPromptEmbedding(
        num_latents=3,
        prompt_length=4,
        hidden_size=8,
        init_strategy="zeros",
    )
    model = SoftPromptCausalLM(base_model, soft_prompt)

    output = model(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        latent_ids=torch.tensor([0]),
        labels=torch.tensor([[-100, 2, 3, -100]]),
        target_latent_ids=torch.tensor([2]),
        latent_positions=torch.tensor([3]),
    )

    assert output[0] is output.loss
    assert output[1].shape == (1, 8, 16)
    assert output.text_loss is not None
    assert output.latent_loss is not None


def test_latent_marker_parser_extracts_last_valid_marker() -> None:
    text = "本文<|emotion|>001<|/emotion|> 後続 <|emotion|>002<|/emotion|>"

    assert parse_latent_id(text, num_latents=4, previous_latent_id=0) == 2


def test_latent_marker_parser_fallbacks() -> None:
    assert (
        parse_latent_id(
            "invalid",
            num_latents=4,
            previous_latent_id=3,
            fallback="previous",
        )
        == 3
    )
    assert parse_latent_id("invalid", num_latents=4, fallback="neutral") == 0
    with pytest.raises(ValueError):
        parse_latent_id("invalid", num_latents=4, fallback="error")


def test_latent_marker_spec_exposes_special_tokens() -> None:
    spec = LatentMarkerSpec("<|emotion|>{latent_id:03d}<|/emotion|>")

    assert spec.special_tokens == ["<|emotion|>", "<|/emotion|>"]
    assert spec.format(7) == "<|emotion|>007<|/emotion|>"


def test_soft_prompt_checkpoint_roundtrip(tmp_path: Path) -> None:
    soft_prompt = SoftPromptEmbedding(
        num_latents=2,
        prompt_length=3,
        hidden_size=5,
        init_strategy="normal",
    )
    model = SoftPromptCausalLM(DummyCausalLM(hidden_size=5), soft_prompt)

    checkpoint = model.save_soft_prompt(tmp_path)
    restored = load_soft_prompt(checkpoint)

    assert restored.weight.shape == (2, 3, 5)
    assert torch.equal(restored.weight, soft_prompt.weight)


def test_latent_head_checkpoint_roundtrip(tmp_path: Path) -> None:
    soft_prompt = SoftPromptEmbedding(
        num_latents=2,
        prompt_length=3,
        hidden_size=5,
        init_strategy="normal",
    )
    model = SoftPromptCausalLM(DummyCausalLM(hidden_size=5), soft_prompt)
    with torch.no_grad():
        model.latent_head.weight.fill_(0.25)

    checkpoint = model.save_latent_head(tmp_path)
    restored = SoftPromptCausalLM(DummyCausalLM(hidden_size=5), soft_prompt)
    assert restored.load_latent_head(checkpoint) is True

    assert torch.equal(restored.latent_head.weight, model.latent_head.weight)
