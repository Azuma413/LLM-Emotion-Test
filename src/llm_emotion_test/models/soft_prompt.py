from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


class SoftPromptEmbedding(nn.Module):
    def __init__(
        self,
        *,
        num_latents: int,
        prompt_length: int,
        hidden_size: int,
        init_strategy: str = "normal",
        token_embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        self.num_latents = num_latents
        self.prompt_length = prompt_length
        self.hidden_size = hidden_size
        self.init_strategy = init_strategy
        self.weight = nn.Parameter(torch.empty(num_latents, prompt_length, hidden_size))
        self.reset_parameters(token_embedding)

    def reset_parameters(self, token_embedding: nn.Embedding | None = None) -> None:
        if self.init_strategy == "zeros":
            nn.init.zeros_(self.weight)
            return
        if self.init_strategy == "mean_token":
            if token_embedding is None:
                raise ValueError("token_embedding is required for init_strategy='mean_token'")
            with torch.no_grad():
                mean = token_embedding.weight.detach().mean(dim=0)
                self.weight.copy_(mean.expand_as(self.weight))
            return
        if self.init_strategy == "normal":
            nn.init.normal_(self.weight, mean=0.0, std=0.02)
            return
        raise ValueError(f"Unsupported soft prompt init_strategy: {self.init_strategy}")

    def forward(self, latent_ids: torch.Tensor) -> torch.Tensor:
        latent_ids = latent_ids.to(device=self.weight.device, dtype=torch.long)
        if torch.any(latent_ids < 0) or torch.any(latent_ids >= self.num_latents):
            raise ValueError("latent_ids contain values outside the configured latent range")
        return self.weight[latent_ids]


class SoftPromptCausalLM(nn.Module):
    """Causal LM wrapper that prepends learnable latent prompts to token embeddings."""

    def __init__(self, base_model: nn.Module, soft_prompt: SoftPromptEmbedding) -> None:
        super().__init__()
        self.base_model = base_model
        self.soft_prompt = soft_prompt

    @property
    def config(self):
        return self.base_model.config

    def get_input_embeddings(self) -> nn.Embedding:
        return self.base_model.get_input_embeddings()

    def resize_token_embeddings(self, size: int):
        return self.base_model.resize_token_embeddings(size)

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        latent_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        if input_ids is None:
            raise ValueError("input_ids is required")
        if latent_ids is None:
            latent_ids = torch.zeros(
                input_ids.shape[0], dtype=torch.long, device=input_ids.device
            )

        token_embeddings = self.get_input_embeddings()(input_ids)
        prompt_embeddings = self.soft_prompt(latent_ids).to(
            device=token_embeddings.device,
            dtype=token_embeddings.dtype,
        )
        inputs_embeds = torch.cat([prompt_embeddings, token_embeddings], dim=1)

        prompt_mask = torch.ones(
            input_ids.shape[0],
            self.soft_prompt.prompt_length,
            dtype=attention_mask.dtype if attention_mask is not None else torch.long,
            device=input_ids.device,
        )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        if labels is not None:
            prompt_labels = torch.full(
                (labels.shape[0], self.soft_prompt.prompt_length),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([prompt_labels, labels], dim=1)

        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        latent_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        if latent_ids is None:
            latent_ids = torch.zeros(
                input_ids.shape[0], dtype=torch.long, device=input_ids.device
            )
        token_embeddings = self.get_input_embeddings()(input_ids)
        prompt_embeddings = self.soft_prompt(latent_ids).to(
            device=token_embeddings.device,
            dtype=token_embeddings.dtype,
        )
        inputs_embeds = torch.cat([prompt_embeddings, token_embeddings], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        prompt_mask = torch.ones(
            input_ids.shape[0],
            self.soft_prompt.prompt_length,
            dtype=attention_mask.dtype,
            device=input_ids.device,
        )
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        return self.base_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    def save_soft_prompt(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_path / "soft_prompt.pt"
        torch.save(
            {
                "state_dict": self.soft_prompt.state_dict(),
                "num_latents": self.soft_prompt.num_latents,
                "prompt_length": self.soft_prompt.prompt_length,
                "hidden_size": self.soft_prompt.hidden_size,
                "init_strategy": self.soft_prompt.init_strategy,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def save_checkpoint(
        self,
        output_dir: str | Path,
        *,
        base_model_id: str,
        tokenizer=None,
        emotion_label_mapping: dict[str, int] | None = None,
    ) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.save_soft_prompt(output_path)
        self.base_model.save_pretrained(
            output_path / "base_or_adapter",
            safe_serialization=False,
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(output_path / "tokenizer")
        metadata = {
            "base_model_id": base_model_id,
            "emotion_label_mapping": emotion_label_mapping or {},
        }
        (output_path / "model_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    def save_pretrained(self, output_dir: str | Path, **kwargs: Any) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.save_soft_prompt(output_path)
        if hasattr(self.base_model, "save_pretrained"):
            kwargs["safe_serialization"] = False
            self.base_model.save_pretrained(output_path / "base_or_adapter", **kwargs)


def load_soft_prompt(checkpoint_path: str | Path) -> SoftPromptEmbedding:
    payload = torch.load(checkpoint_path, map_location="cpu")
    soft_prompt = SoftPromptEmbedding(
        num_latents=int(payload["num_latents"]),
        prompt_length=int(payload["prompt_length"]),
        hidden_size=int(payload["hidden_size"]),
        init_strategy=str(payload.get("init_strategy", "normal")),
    )
    soft_prompt.load_state_dict(payload["state_dict"])
    return soft_prompt
