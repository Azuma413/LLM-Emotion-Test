from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import yaml
from torch.utils.data import Dataset
from transformers import Trainer, set_seed

from llm_emotion_test.config import ExperimentConfig
from llm_emotion_test.data.distill import prepare_distillation_dataset
from llm_emotion_test.models.loader import build_soft_prompt_model, save_model_checkpoint
from llm_emotion_test.training.sft import (
    EmotionSFTDataCollator,
    build_training_arguments,
    generate_sft_samples,
    latent_marker_accuracy,
    load_sft_records,
    write_jsonl,
)


def build_distill_student_prompt(record: Mapping[str, Any]) -> str:
    return "ユーザー入力:\n" f"{record['input_text']}\n\n" "応答:\n"


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


def train_distill(config: ExperimentConfig) -> dict[str, Any]:
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

    model, tokenizer = build_soft_prompt_model(config)
    collator = EmotionSFTDataCollator(
        tokenizer=tokenizer,
        max_seq_length=config.training.max_seq_length,
    )
    trainer = Trainer(
        model=model,
        args=build_training_arguments(config),
        train_dataset=DistillationStudentDataset(train_records),
        eval_dataset=DistillationStudentDataset(eval_records),
        data_collator=collator,
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
        "sample_latent_marker_accuracy": latent_marker_accuracy(samples),
        "kl_divergence_weight": config.distillation.kl_divergence_weight,
    }
    write_jsonl([metrics], config.output.metrics_path)
    return metrics
