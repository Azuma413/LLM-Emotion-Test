from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, set_seed

from llm_emotion_test.config import ExperimentConfig
from llm_emotion_test.data.distill import prepare_distillation_dataset
from llm_emotion_test.models.loader import (
    build_soft_prompt_model,
    load_soft_prompt_model_from_checkpoint,
    require_cuda_if_configured,
    resolve_torch_dtype,
    save_model_checkpoint,
)
from llm_emotion_test.training.sft import (
    EmotionSFTDataCollator,
    LatentLossLoggingTrainer,
    build_training_arguments,
    generate_sft_samples,
    latent_marker_accuracy,
    latent_accuracy,
    load_sft_records,
    write_jsonl,
)


def build_distill_student_prompt(record: Mapping[str, Any]) -> str:
    return str(record["input_text"])


class DistillationStudentDataset(Dataset):
    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.records = [dict(record) for record in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "prompt": build_distill_student_prompt(record),
            "target": str(record["target_text"]),
            "latent_id": int(record["input_latent_id"]),
            "target_latent_id": int(record["target_latent_id"]),
        }


class DistillationTrainer(LatentLossLoggingTrainer):
    def __init__(self, *args, teacher_model=None, kl_weight: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.kl_weight = kl_weight
        self.last_kl_loss: float | None = None
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for parameter in self.teacher_model.parameters():
                parameter.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        self._record_latent_losses(outputs)
        loss = outputs.loss
        if self.teacher_model is not None and self.kl_weight > 0.0:
            kl_loss = self._kl_loss(model, outputs, inputs)
            self.last_kl_loss = float(kl_loss.detach().cpu())
            loss = loss + self.kl_weight * kl_loss
        return (loss, outputs) if return_outputs else loss

    def _kl_loss(self, model, student_outputs, inputs) -> torch.Tensor:
        teacher_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
        }
        teacher_device = next(self.teacher_model.parameters()).device
        teacher_inputs = {
            key: value.to(teacher_device) if value is not None else None
            for key, value in teacher_inputs.items()
        }
        teacher_inputs = {key: value for key, value in teacher_inputs.items() if value is not None}
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs)

        prompt_length = getattr(model.soft_prompt, "prompt_length", 0)
        student_logits = student_outputs.logits[:, prompt_length:, :]
        teacher_logits = teacher_outputs.logits.to(student_logits.device)
        labels = inputs["labels"].to(student_logits.device)
        mask = labels != -100
        if not torch.any(mask):
            return torch.zeros((), dtype=student_logits.dtype, device=student_logits.device)

        vocab_size = min(student_logits.shape[-1], teacher_logits.shape[-1])
        student_log_probs = F.log_softmax(student_logits[..., :vocab_size], dim=-1)
        teacher_probs = F.softmax(teacher_logits[..., :vocab_size], dim=-1)
        token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        return token_kl[mask].mean()


def train_distill(config: ExperimentConfig) -> dict[str, Any]:
    require_cuda_if_configured(config)
    set_seed(config.runtime.seed)
    config.output.run_dir.mkdir(parents=True, exist_ok=True)
    config.output.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config.output.config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    distill_stats = prepare_distillation_dataset(config)
    train_records = load_sft_records(config.data.processed_path, split="train")
    eval_records = load_sft_records(config.data.processed_path, split="validation")
    if not train_records:
        raise ValueError("Distillation data has no train split records")
    if not eval_records:
        eval_records = train_records[: min(len(train_records), config.training.eval_max_samples)]

    if config.distillation.student_checkpoint_dir is not None:
        model, tokenizer = load_soft_prompt_model_from_checkpoint(
            config.distillation.student_checkpoint_dir,
            config,
        )
    else:
        model, tokenizer = build_soft_prompt_model(config)
    collator = EmotionSFTDataCollator(
        tokenizer=tokenizer,
        max_seq_length=config.training.max_seq_length,
        marker_template=config.soft_prompt.latent_marker_template,
        latent_training_mode=config.latent_training.mode,
        anchor_token=config.latent_training.anchor_token,
    )
    teacher_model = build_kl_teacher_model(config, tokenizer) if (
        config.distillation.kl_divergence_weight > 0.0
    ) else None
    trainer = DistillationTrainer(
        model=model,
        args=build_training_arguments(config),
        train_dataset=DistillationStudentDataset(train_records),
        eval_dataset=DistillationStudentDataset(eval_records),
        data_collator=collator,
        teacher_model=teacher_model,
        kl_weight=config.distillation.kl_divergence_weight,
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    final_checkpoint = save_model_checkpoint(
        model,
        tokenizer,
        config,
        config.output.checkpoints_dir / "final",
    )
    samples = generate_sft_samples(
        model,
        tokenizer,
        eval_records[: config.training.sample_count],
        config=config,
    )
    write_jsonl(samples, config.output.samples_path)

    metrics = {
        "distillation": distill_stats,
        "train": dict(train_result.metrics),
        "eval": dict(eval_metrics),
        "final_checkpoint": str(final_checkpoint),
        "sample_latent_accuracy": latent_accuracy(samples),
        "sample_latent_marker_accuracy": latent_marker_accuracy(samples),
        "kl_divergence_weight": config.distillation.kl_divergence_weight,
        "kl_loss": trainer.last_kl_loss,
        "text_loss": trainer.last_text_loss,
        "latent_loss": trainer.last_latent_loss,
    }
    write_jsonl([metrics], config.output.metrics_path)
    return metrics


def build_kl_teacher_model(config: ExperimentConfig, tokenizer):
    kwargs: dict[str, Any] = {
        "trust_remote_code": config.distillation.teacher_trust_remote_code,
    }
    if config.distillation.teacher_device_map is not None:
        kwargs["device_map"] = config.distillation.teacher_device_map
    dtype = resolve_torch_dtype(config.distillation.teacher_torch_dtype)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    teacher = AutoModelForCausalLM.from_pretrained(
        config.distillation.teacher_model,
        **kwargs,
    )
    teacher.resize_token_embeddings(len(tokenizer))
    return teacher
