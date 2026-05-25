from __future__ import annotations

from torch import nn
from transformers import TrainingArguments

from llm_emotion_test.training.sft import (
    EmotionSFTDataCollator,
    EmotionSFTDataset,
    LatentLossLoggingTrainer,
    build_training_arguments,
    build_sft_prompt,
    latent_marker_accuracy,
)
from llm_emotion_test.config import ExperimentConfig


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = -1

    def __call__(self, text, add_special_tokens=False, **_kwargs):
        return {"input_ids": [ord(char) % 251 + 2 for char in text]}

    def convert_tokens_to_ids(self, token):
        if token == "<|latent_pred|>":
            return 999
        return self.unk_token_id

    def save_pretrained(self, output_dir):
        return None


class SharedWeightModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(4, 3)
        self.head = nn.Linear(3, 4, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, input_ids=None, **_kwargs):
        hidden = self.embedding(input_ids)
        logits = self.head(hidden)
        return {"loss": logits.sum() * 0, "logits": logits}


def test_sft_dataset_builds_prompt_and_target() -> None:
    record = {
        "input_text": "今日は疲れ",
        "target_text": "休みましょう\n<|emotion|>001<|/emotion|>",
        "input_latent_id": 1,
        "target_latent_id": 1,
    }

    item = EmotionSFTDataset([record])[0]

    assert item["prompt"] == "今日は疲れ"
    assert "ユーザー入力" not in item["prompt"]
    assert "現在のlatent ID" not in item["prompt"]
    assert "<|emotion|>" not in item["prompt"]
    assert item["target"] == record["target_text"]
    assert item["latent_id"] == 1


def test_sft_collator_masks_prompt_loss() -> None:
    record = {
        "input_text": "input",
        "target_text": "target",
        "input_latent_id": 2,
        "target_latent_id": 3,
    }
    item = EmotionSFTDataset([record])[0]
    collator = EmotionSFTDataCollator(TinyTokenizer(), max_seq_length=128)

    batch = collator([item])

    prompt_length = len(TinyTokenizer()(build_sft_prompt(record))["input_ids"])
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["latent_ids"].tolist() == [2]
    assert batch["target_latent_ids"].tolist() == [3]
    assert batch["latent_positions"].tolist() == [batch["attention_mask"][0].sum().item() - 1]
    assert batch["labels"][0, :prompt_length].tolist() == [-100] * prompt_length
    assert all(label != -100 for label in batch["labels"][0, prompt_length:-1].tolist())
    assert batch["labels"][0, -1].item() == -100


def test_sft_collator_strips_terminal_marker_from_labels() -> None:
    record = {
        "input_text": "input",
        "target_text": "target<|emotion|>003<|/emotion|>",
        "input_latent_id": 2,
        "target_latent_id": 3,
    }
    item = EmotionSFTDataset([record])[0]
    tokenizer = TinyTokenizer()
    collator = EmotionSFTDataCollator(tokenizer, max_seq_length=128)

    batch = collator([item])

    marker_ids = tokenizer("<|emotion|>003<|/emotion|>")["input_ids"]
    label_ids = [int(value) for value in batch["labels"][0].tolist() if int(value) != -100]
    assert label_ids[-1] == tokenizer.eos_token_id
    assert not any(
        label_ids[index : index + len(marker_ids)] == marker_ids
        for index in range(len(label_ids))
    )


def test_latent_marker_accuracy() -> None:
    samples = [
        {"predicted_latent_id": 1, "target_latent_id": 1},
        {"predicted_latent_id": 2, "target_latent_id": 1},
    ]

    assert latent_marker_accuracy(samples) == 0.5


def test_training_arguments_use_regular_ddp_graph() -> None:
    args = build_training_arguments(
        ExperimentConfig.model_validate(
            {
                "training": {
                    "max_steps": 1,
                    "report_to": "none",
                },
            }
        )
    )

    assert args.ddp_find_unused_parameters is False
    assert args.ddp_static_graph is False


def test_trainer_checkpoint_save_allows_tied_weights(tmp_path) -> None:
    trainer = LatentLossLoggingTrainer(
        model=SharedWeightModel(),
        args=TrainingArguments(output_dir=str(tmp_path), report_to=[]),
        data_collator=EmotionSFTDataCollator(TinyTokenizer(), max_seq_length=16),
    )

    checkpoint_dir = tmp_path / "checkpoint-1"
    trainer._save(str(checkpoint_dir))

    assert (checkpoint_dir / "pytorch_model.bin").exists()
    assert not (checkpoint_dir / "model.safetensors").exists()
