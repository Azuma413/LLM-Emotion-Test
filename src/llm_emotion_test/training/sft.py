from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, set_seed

from llm_emotion_test.config import ExperimentConfig
from llm_emotion_test.data.wrime import prepare_wrime_dataset
from llm_emotion_test.models.latent import LatentMarkerSpec, strip_terminal_latent_marker
from llm_emotion_test.models.loader import build_soft_prompt_model, save_model_checkpoint


def load_sft_records(path: str | Path, *, split: str | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if split is None or record.get("split") == split:
                records.append(record)
            if not isinstance(record.get("input_latent_id"), int):
                raise ValueError(f"Invalid input_latent_id at line {line_number}")
            if not isinstance(record.get("target_latent_id"), int):
                raise ValueError(f"Invalid target_latent_id at line {line_number}")
    return records


def build_sft_prompt(record: Mapping[str, Any]) -> str:
    return str(record["input_text"])


def build_sft_target(record: Mapping[str, Any]) -> str:
    return str(record["target_text"])


class EmotionSFTDataset(Dataset):
    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.records = [dict(record) for record in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "prompt": build_sft_prompt(record),
            "target": build_sft_target(record),
            "latent_id": int(record["input_latent_id"]),
            "target_latent_id": int(record["target_latent_id"]),
        }


@dataclass
class EmotionSFTDataCollator:
    tokenizer: Any
    max_seq_length: int
    marker_template: str = "<|emotion|>{latent_id:03d}<|/emotion|>"
    latent_training_mode: str = "regression"
    anchor_token: str = "<|latent_pred|>"

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        encoded_rows = [self._encode_feature(feature) for feature in features]
        max_length = max(len(row["input_ids"]) for row in encoded_rows)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        latent_positions: list[int] = []
        for row in encoded_rows:
            pad_length = max_length - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_token_id] * pad_length)
            attention_mask.append([1] * len(row["input_ids"]) + [0] * pad_length)
            labels.append(row["labels"] + [-100] * pad_length)
            latent_positions.append(row["latent_position"])

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "latent_ids": torch.tensor(
                [int(feature["latent_id"]) for feature in features], dtype=torch.long
            ),
        }
        if self.latent_training_mode != "marker_ce":
            batch["target_latent_ids"] = torch.tensor(
                [int(feature["target_latent_id"]) for feature in features],
                dtype=torch.long,
            )
            batch["latent_positions"] = torch.tensor(latent_positions, dtype=torch.long)
        return batch

    def _encode_feature(self, feature: Mapping[str, Any]) -> dict[str, list[int]]:
        prompt_ids = self.tokenizer(
            str(feature["prompt"]),
            add_special_tokens=False,
        )["input_ids"]
        target_text = str(feature["target"])
        if self.latent_training_mode == "regression":
            target_text = strip_terminal_latent_marker(
                target_text,
                marker_template=self.marker_template,
            )
        target_ids = self.tokenizer(
            target_text,
            add_special_tokens=False,
        )["input_ids"]
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            target_ids = target_ids + [int(eos_token_id)]

        reserve_anchor = self.latent_training_mode != "marker_ce"
        prompt_ids, target_ids = truncate_prompt_and_target(
            prompt_ids,
            target_ids,
            self.max_seq_length - 1 if reserve_anchor else self.max_seq_length,
        )
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        latent_position = len(input_ids)
        if reserve_anchor:
            anchor_token_id = self._anchor_token_id()
            input_ids = input_ids + [anchor_token_id]
            labels = labels + [-100]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "latent_position": latent_position,
        }

    def _anchor_token_id(self) -> int:
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            token_id = self.tokenizer.convert_tokens_to_ids(self.anchor_token)
            unknown_id = getattr(self.tokenizer, "unk_token_id", None)
            if token_id is not None and token_id != unknown_id:
                return int(token_id)
        token_ids = self.tokenizer(self.anchor_token, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(
                f"Latent anchor token must tokenize to one id, got {len(token_ids)} ids"
            )
        return int(token_ids[0])


def truncate_prompt_and_target(
    prompt_ids: Sequence[int],
    target_ids: Sequence[int],
    max_seq_length: int,
) -> tuple[list[int], list[int]]:
    prompt = list(prompt_ids)
    target = list(target_ids)
    if len(target) >= max_seq_length:
        return [], target[:max_seq_length]
    available_prompt_length = max_seq_length - len(target)
    if len(prompt) > available_prompt_length:
        prompt = prompt[-available_prompt_length:]
    return prompt, target


def build_training_arguments(config: ExperimentConfig) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(config.output.checkpoints_dir),
        "per_device_train_batch_size": config.training.batch_size,
        "per_device_eval_batch_size": config.training.batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "num_train_epochs": config.training.num_train_epochs,
        "logging_steps": config.training.logging_steps,
        "warmup_steps": config.training.warmup_steps,
        "weight_decay": config.training.weight_decay,
        "save_strategy": "steps" if config.training.save_steps else "no",
        "report_to": [] if config.training.report_to == "none" else [config.training.report_to],
        "remove_unused_columns": False,
        "bf16": config.training.precision == "bf16",
        "fp16": config.training.precision == "fp16",
        "ddp_find_unused_parameters": False,
        "ddp_static_graph": True,
    }
    if config.training.max_steps is not None:
        kwargs["max_steps"] = config.training.max_steps
    if config.training.save_steps is not None:
        kwargs["save_steps"] = config.training.save_steps
    eval_steps = config.training.eval_steps
    if eval_steps is not None:
        kwargs["eval_steps"] = eval_steps
    kwargs["eval_strategy" if _training_arguments_accepts_eval_strategy() else "evaluation_strategy"] = (
        "steps" if eval_steps else "no"
    )
    return TrainingArguments(**kwargs)


def _training_arguments_accepts_eval_strategy() -> bool:
    return "eval_strategy" in TrainingArguments.__init__.__code__.co_varnames


class LatentLossLoggingTrainer(Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_text_loss: float | None = None
        self.last_latent_loss: float | None = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        self._record_latent_losses(outputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def _record_latent_losses(self, outputs) -> None:
        text_loss = getattr(outputs, "text_loss", None)
        latent_loss = getattr(outputs, "latent_loss", None)
        if text_loss is not None:
            self.last_text_loss = float(text_loss.detach().cpu())
        if latent_loss is not None:
            self.last_latent_loss = float(latent_loss.detach().cpu())

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if self.last_text_loss is not None:
            logs["text_loss"] = self.last_text_loss
        if self.last_latent_loss is not None:
            logs["latent_loss"] = self.last_latent_loss
        super().log(logs, *args, **kwargs)


def train_sft(config: ExperimentConfig) -> dict[str, Any]:
    set_seed(config.runtime.seed)
    config.output.run_dir.mkdir(parents=True, exist_ok=True)
    config.output.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config.output.config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    if not config.data.processed_path.exists():
        prepare_wrime_dataset(config)

    train_records = load_sft_records(config.data.processed_path, split="train")
    eval_records = load_sft_records(config.data.processed_path, split="validation")
    if not train_records:
        raise ValueError("SFT training data has no train split records")
    if not eval_records:
        eval_records = train_records[: min(len(train_records), config.training.eval_max_samples)]

    model, tokenizer = build_soft_prompt_model(config, device_map=None)
    collator = EmotionSFTDataCollator(
        tokenizer=tokenizer,
        max_seq_length=config.training.max_seq_length,
        marker_template=config.soft_prompt.latent_marker_template,
        latent_training_mode=config.latent_training.mode,
        anchor_token=config.latent_training.anchor_token,
    )
    trainer = LatentLossLoggingTrainer(
        model=model,
        args=build_training_arguments(config),
        train_dataset=EmotionSFTDataset(train_records),
        eval_dataset=EmotionSFTDataset(eval_records),
        data_collator=collator,
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    final_checkpoint = config.output.checkpoints_dir / "final"
    samples: list[dict[str, Any]] = []
    if trainer.is_world_process_zero():
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        final_checkpoint = save_model_checkpoint(
            unwrapped_model,
            tokenizer,
            config,
            final_checkpoint,
        )
        samples = generate_sft_samples(
            unwrapped_model,
            tokenizer,
            eval_records[: config.training.sample_count],
            config=config,
        )
        write_jsonl(samples, config.output.samples_path)

    metrics = {
        "train": dict(train_result.metrics),
        "eval": dict(eval_metrics),
        "final_checkpoint": str(final_checkpoint),
        "sample_latent_accuracy": latent_accuracy(samples),
        "sample_latent_marker_accuracy": latent_accuracy(samples),
        "text_loss": trainer.last_text_loss,
        "latent_loss": trainer.last_latent_loss,
    }
    if trainer.is_world_process_zero():
        write_jsonl([metrics], config.output.metrics_path)
    return metrics


@torch.no_grad()
def generate_sft_samples(
    model,
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    *,
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    if not records:
        return []
    model.eval()
    samples: list[dict[str, Any]] = []
    device = next(model.parameters()).device
    for record in records:
        prompt = build_sft_prompt(record)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        latent_id = int(record["input_latent_id"])
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_ids=torch.tensor([latent_id], device=device, dtype=torch.long),
            max_new_tokens=config.training.generation_max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        predicted_latent_id, latent_distance = predict_latent_from_text(
            model,
            tokenizer,
            prompt + generated_text,
            input_latent_id=latent_id,
            anchor_token=config.latent_training.anchor_token,
        )
        generated_with_marker = (
            generated_text
            + LatentMarkerSpec(config.soft_prompt.latent_marker_template).format(
                predicted_latent_id
            )
        )
        samples.append(
            {
                "prompt": prompt,
                "target_text": str(record["target_text"]),
                "generated_text": generated_with_marker,
                "generated_response_text": generated_text,
                "input_latent_id": latent_id,
                "target_latent_id": int(record["target_latent_id"]),
                "predicted_latent_id": predicted_latent_id,
                "latent_distance": latent_distance,
            }
        )
    return samples


@torch.no_grad()
def predict_latent_from_text(
    model,
    tokenizer,
    text: str,
    *,
    input_latent_id: int,
    anchor_token: str,
) -> tuple[int, float]:
    device = next(model.parameters()).device
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    anchor_id = _token_id(tokenizer, anchor_token)
    input_ids = torch.tensor([list(encoded) + [anchor_id]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    latent_positions = torch.tensor([len(encoded)], dtype=torch.long, device=device)
    latent_ids = torch.tensor([input_latent_id], dtype=torch.long, device=device)
    predicted, distances = model.predict_latent(
        input_ids=input_ids,
        attention_mask=attention_mask,
        latent_ids=latent_ids,
        latent_positions=latent_positions,
    )
    return int(predicted[0].detach().cpu()), float(distances[0].detach().cpu())


def _token_id(tokenizer, token: str) -> int:
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        unknown_id = getattr(tokenizer, "unk_token_id", None)
        if token_id is not None and token_id != unknown_id:
            return int(token_id)
    token_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
    if len(token_ids) != 1:
        raise ValueError(f"Token must tokenize to one id, got {len(token_ids)} ids")
    return int(token_ids[0])


def latent_accuracy(samples: Sequence[Mapping[str, Any]]) -> float | None:
    if not samples:
        return None
    correct = sum(
        int(sample["predicted_latent_id"]) == int(sample["target_latent_id"])
        for sample in samples
    )
    return correct / len(samples)


latent_marker_accuracy = latent_accuracy


def write_jsonl(records: Sequence[Mapping[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            file.write("\n")
    return path
