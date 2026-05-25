from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


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

    def __init__(
        self,
        base_model: nn.Module,
        soft_prompt: SoftPromptEmbedding,
        *,
        latent_loss_weight: float = 1.0,
        latent_target: str = "soft_prompt_mean",
        detach_latent_target: bool = True,
        normalize_latent_loss: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.soft_prompt = soft_prompt
        self.latent_loss_weight = latent_loss_weight
        self.latent_target = latent_target
        self.detach_latent_target = detach_latent_target
        self.normalize_latent_loss = normalize_latent_loss
        target_size = (
            soft_prompt.hidden_size
            if latent_target == "soft_prompt_mean"
            else soft_prompt.prompt_length * soft_prompt.hidden_size
        )
        self.latent_head = nn.Linear(soft_prompt.hidden_size, target_size)

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
        target_latent_ids: torch.Tensor | None = None,
        latent_positions: torch.Tensor | None = None,
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

        needs_latent = target_latent_ids is not None and latent_positions is not None
        if needs_latent:
            kwargs["output_hidden_states"] = True
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        if not needs_latent:
            return outputs

        latent_loss, pred_latent = self.compute_latent_loss(
            outputs=outputs,
            inputs_embeds=inputs_embeds,
            target_latent_ids=target_latent_ids,
            latent_positions=latent_positions,
        )
        text_loss = outputs.loss
        total_loss = latent_loss * self.latent_loss_weight
        if text_loss is not None:
            total_loss = text_loss + total_loss
        return _replace_output_fields(
            outputs,
            loss=total_loss,
            text_loss=text_loss,
            latent_loss=latent_loss,
            pred_latent=pred_latent,
        )

    def compute_latent_loss(
        self,
        *,
        outputs,
        inputs_embeds: torch.Tensor,
        target_latent_ids: torch.Tensor,
        latent_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None:
            final_hidden = hidden_states[-1]
        else:
            final_hidden = inputs_embeds
        shifted_positions = latent_positions.to(final_hidden.device) + self.soft_prompt.prompt_length
        batch_indices = torch.arange(final_hidden.shape[0], device=final_hidden.device)
        anchor_hidden = final_hidden[batch_indices, shifted_positions]
        pred_latent = self.latent_head(anchor_hidden)
        target = self.soft_prompt(target_latent_ids).to(
            device=pred_latent.device,
            dtype=pred_latent.dtype,
        )
        if self.latent_target == "soft_prompt_mean":
            target = target.mean(dim=1)
        elif self.latent_target == "soft_prompt_flatten":
            target = target.flatten(start_dim=1)
        else:
            raise ValueError(f"Unsupported latent target: {self.latent_target}")
        if self.detach_latent_target:
            target = target.detach()
        pred_for_loss = pred_latent
        target_for_loss = target
        if self.normalize_latent_loss:
            pred_for_loss = F.normalize(pred_for_loss, dim=-1)
            target_for_loss = F.normalize(target_for_loss, dim=-1)
        return F.mse_loss(pred_for_loss, target_for_loss), pred_latent

    @torch.no_grad()
    def predict_latent(
        self,
        *,
        input_ids: torch.Tensor,
        latent_ids: torch.Tensor,
        latent_positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self(
            input_ids=input_ids,
            latent_ids=latent_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        final_hidden = outputs.hidden_states[-1]
        shifted_positions = latent_positions.to(final_hidden.device) + self.soft_prompt.prompt_length
        batch_indices = torch.arange(final_hidden.shape[0], device=final_hidden.device)
        pred_latent = self.latent_head(final_hidden[batch_indices, shifted_positions])
        codebook = self.soft_prompt.weight.to(device=pred_latent.device, dtype=pred_latent.dtype)
        if self.latent_target == "soft_prompt_mean":
            targets = codebook.mean(dim=1)
        elif self.latent_target == "soft_prompt_flatten":
            targets = codebook.flatten(start_dim=1)
        else:
            raise ValueError(f"Unsupported latent target: {self.latent_target}")
        pred_compare = F.normalize(pred_latent, dim=-1) if self.normalize_latent_loss else pred_latent
        target_compare = F.normalize(targets, dim=-1) if self.normalize_latent_loss else targets
        distances = torch.cdist(pred_compare, target_compare)
        return distances.argmin(dim=-1), distances.min(dim=-1).values

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

    def save_latent_head(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_path / "latent_head.pt"
        torch.save(
            {
                "state_dict": self.latent_head.state_dict(),
                "latent_loss_weight": self.latent_loss_weight,
                "latent_target": self.latent_target,
                "detach_latent_target": self.detach_latent_target,
                "normalize_latent_loss": self.normalize_latent_loss,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def load_latent_head(self, checkpoint_path: str | Path) -> bool:
        path = Path(checkpoint_path)
        if not path.exists():
            return False
        payload = torch.load(path, map_location="cpu")
        self.latent_head.load_state_dict(payload["state_dict"])
        return True

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
        self.save_latent_head(output_path)
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
        self.save_latent_head(output_path)
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


def _replace_output_fields(outputs, **fields: Any):
    for key, value in fields.items():
        setattr(outputs, key, value)
    return outputs
