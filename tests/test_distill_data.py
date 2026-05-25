from __future__ import annotations

import json
from pathlib import Path

from llm_emotion_test.config import ExperimentConfig
from llm_emotion_test.data.distill import (
    build_teacher_instruction,
    build_teacher_prompt,
    prepare_distillation_dataset,
)
from llm_emotion_test.training.distill import build_distill_student_prompt


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")


def test_teacher_prompt_includes_japanese_emotion_instruction() -> None:
    instruction = build_teacher_instruction("anger")
    prompt = build_teacher_prompt(
        "今日は疲れた",
        instruction,
    )

    assert "怒りっぽい" in instruction
    assert "日本語で自然に返答" in prompt
    assert "入力:" in prompt
    assert "<|emotion|>" not in prompt
    assert "latent marker" not in prompt
    assert "ユーザー入力" not in prompt


def test_prepare_distillation_dataset_uses_cache_and_writes_student_sft(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    write_jsonl(
        source_path,
        [
            {
                "input_text": "今日は疲れた\n<|emotion|>001<|/emotion|>",
                "target_text": "今日は疲れた\n<|emotion|>001<|/emotion|>",
                "emotion_labels": {"sadness": 1},
                "input_latent_id": 1,
                "target_latent_id": 1,
                "split": "train",
                "source_text": "今日は疲れた。休みたい",
            }
        ],
    )
    config = ExperimentConfig.model_validate(
        {
            "data": {
                "processed_dir": tmp_path,
                "processed_filename": "student.jsonl",
                "max_samples": 1,
            },
            "output": {"root": tmp_path / "runs", "run_id": "distill"},
            "training": {"report_to": "none"},
            "distillation": {"source_data_path": source_path},
        }
    )

    stats = prepare_distillation_dataset(
        config,
        generator=lambda prompts: [
            "少し悲しいですが、休む時間を取りましょう。\n<|emotion|>001<|/emotion|>"
            for _prompt in prompts
        ],
    )

    assert stats["num_distill_records"] == 1
    student_records = [
        json.loads(line) for line in (tmp_path / "student.jsonl").read_text().splitlines()
    ]
    assert student_records[0]["input_text"] == "今日は疲れた"
    assert student_records[0]["input_latent_id"] == 1
    assert student_records[0]["target_text"].endswith("<|emotion|>001<|/emotion|>")
    assert student_records[0]["target_text"].count("<|emotion|>") == 1
    assert build_distill_student_prompt(student_records[0]) == "今日は疲れた"

    cache_records = [
        json.loads(line)
        for line in (config.output.run_dir / "teacher_generations.jsonl").read_text().splitlines()
    ]
    assert "<|emotion|>" not in cache_records[0]["teacher_output_text"]
    distill_records = [
        json.loads(line)
        for line in (config.output.run_dir / "distillation_data.jsonl").read_text().splitlines()
    ]
    assert "<|emotion|>" not in distill_records[0]["teacher_output_text"]

    cached_stats = prepare_distillation_dataset(
        config,
        generator=lambda _prompts: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    assert cached_stats["num_distill_records"] == 1
