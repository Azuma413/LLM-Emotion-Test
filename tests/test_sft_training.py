from __future__ import annotations

from llm_emotion_test.training.sft import (
    EmotionSFTDataCollator,
    EmotionSFTDataset,
    build_sft_prompt,
    latent_marker_accuracy,
)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False, **_kwargs):
        return {"input_ids": [ord(char) % 251 + 2 for char in text]}


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
    assert batch["labels"][0, :prompt_length].tolist() == [-100] * prompt_length
    assert all(label != -100 for label in batch["labels"][0, prompt_length:].tolist())


def test_latent_marker_accuracy() -> None:
    samples = [
        {"predicted_latent_id": 1, "target_latent_id": 1},
        {"predicted_latent_id": 2, "target_latent_id": 1},
    ]

    assert latent_marker_accuracy(samples) == 0.5
