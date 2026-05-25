from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_emotion_test.config import ExperimentConfig
from llm_emotion_test.data.wrime import build_label_to_id, format_latent_marker
from llm_emotion_test.models.loader import resolve_torch_dtype


TeacherGenerator = Callable[[Sequence[str]], Sequence[str]]


INSTRUCTION_TEMPLATES: dict[str, str] = {
    "joy": "明るく嬉しそうに、相手に寄り添って返答してください。",
    "sadness": "悲しげに、しかし協力的に返答してください。",
    "anticipation": "期待感をにじませながら、前向きに返答してください。",
    "surprise": "少し驚いた調子で、自然に返答してください。",
    "anger": "怒りっぽい調子を含めつつ、攻撃的になりすぎず返答してください。",
    "fear": "不安そうな調子を含めつつ、慎重に返答してください。",
    "disgust": "嫌悪感を少し含めつつ、礼儀を保って返答してください。",
    "trust": "信頼感と安心感が伝わるように返答してください。",
}


@dataclass(frozen=True)
class DistillationRecord:
    base_input_text: str
    teacher_instruction: str
    teacher_output_text: str
    emotion_label: str
    student_input_latent_id: int
    student_target_latent_id: int
    split: str

    def as_json(self) -> dict[str, Any]:
        return {
            "base_input_text": self.base_input_text,
            "teacher_instruction": self.teacher_instruction,
            "teacher_output_text": self.teacher_output_text,
            "emotion_label": self.emotion_label,
            "student_input_latent_id": self.student_input_latent_id,
            "student_target_latent_id": self.student_target_latent_id,
            "split": self.split,
        }

    def as_student_sft_record(self, *, marker_template: str) -> dict[str, Any]:
        return {
            "input_text": self.base_input_text,
            "target_text": append_latent_marker(
                self.teacher_output_text,
                latent_id=self.student_target_latent_id,
                marker_template=marker_template,
            ),
            "emotion_labels": {self.emotion_label: 1},
            "input_latent_id": self.student_input_latent_id,
            "target_latent_id": self.student_target_latent_id,
            "split": self.split,
        }


def build_teacher_instruction(emotion_label: str) -> str:
    return INSTRUCTION_TEMPLATES.get(
        emotion_label,
        f"{emotion_label} の感情が伝わるように、日本語で返答してください。",
    )


def build_teacher_prompt(
    base_input_text: str,
    teacher_instruction: str,
) -> str:
    return (
        "あなたは日本語で自然に返答するアシスタントです。\n"
        f"感情表現の指示: {teacher_instruction}\n\n"
        f"入力:\n{base_input_text}"
    )


def prepare_distillation_dataset(
    config: ExperimentConfig,
    *,
    generator: TeacherGenerator | None = None,
) -> dict[str, Any]:
    source_path = config.distillation.source_data_path
    if not source_path.exists():
        raise FileNotFoundError(f"Distillation source data does not exist: {source_path}")

    source_records = load_jsonl(source_path)
    if config.data.max_samples is not None:
        source_records = source_records[: config.data.max_samples]

    label_to_id = build_label_to_id(
        config.data.emotion_labels,
        config.data.representative_label_map,
    )
    cache_path = resolve_teacher_cache_path(config)
    cached = {} if config.distillation.overwrite_cache else load_teacher_cache(cache_path)

    rows_to_generate: list[dict[str, Any]] = []
    prompts: list[str] = []
    for source in source_records:
        request = build_teacher_request(source, config, label_to_id)
        cache_key = teacher_cache_key(request)
        if cache_key in cached:
            continue
        rows_to_generate.append({**request, "cache_key": cache_key})
        prompts.append(str(request["teacher_prompt"]))

    if rows_to_generate:
        active_generator = generator or HuggingFaceTeacherGenerator(config)
        generated_outputs = list(
            generate_in_batches(
                active_generator,
                prompts,
                batch_size=config.distillation.teacher_batch_size,
            )
        )
        for request, output in zip(rows_to_generate, generated_outputs, strict=True):
            raw_cached_row = {
                "base_input_text": request["base_input_text"],
                "teacher_instruction": request["teacher_instruction"],
                "teacher_output_text": output,
                "emotion_label": request["emotion_label"],
                "student_input_latent_id": request["student_input_latent_id"],
                "student_target_latent_id": request["student_target_latent_id"],
                "split": request["split"],
            }
            normalized = normalize_teacher_record(raw_cached_row, config)
            if normalized is None:
                continue
            cached[str(request["cache_key"])] = normalized.as_json()
        write_teacher_cache(cached.values(), cache_path)

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    dropped = 0
    for source in source_records:
        request = build_teacher_request(source, config, label_to_id)
        cached_row = cached.get(teacher_cache_key(request))
        if cached_row is None:
            dropped += 1
            continue
        record = normalize_teacher_record(cached_row, config)
        if record is None:
            dropped += 1
            continue
        dedupe_key = (
            record.base_input_text,
            record.emotion_label,
            record.teacher_output_text,
        )
        if config.distillation.deduplicate and dedupe_key in seen:
            dropped += 1
            continue
        seen.add(dedupe_key)
        records.append(record.as_json())

    output_path = write_jsonl(records, resolve_distill_data_path(config))
    student_path = write_jsonl(
        [
            DistillationRecord(**record).as_student_sft_record(
                marker_template=config.soft_prompt.latent_marker_template,
            )
            for record in records
        ],
        config.data.processed_path,
    )
    return {
        "output_path": str(output_path),
        "student_data_path": str(student_path),
        "teacher_cache_path": str(cache_path),
        "num_source_records": len(source_records),
        "num_distill_records": len(records),
        "num_dropped_records": dropped,
    }


def build_teacher_request(
    source: Mapping[str, Any],
    config: ExperimentConfig,
    label_to_id: Mapping[str, int],
) -> dict[str, Any]:
    latent_id = int(source["input_latent_id"])
    emotion_label = label_for_latent_id(latent_id, label_to_id)
    target_latent_id = int(source.get("target_latent_id", latent_id))
    base_input_text = strip_latent_markers(
        str(source["input_text"]),
        marker_template=config.soft_prompt.latent_marker_template,
    )
    instruction = build_teacher_instruction(emotion_label)
    return {
        "base_input_text": base_input_text,
        "teacher_instruction": instruction,
        "teacher_prompt": build_teacher_prompt(
            base_input_text,
            instruction,
        ),
        "emotion_label": emotion_label,
        "student_input_latent_id": latent_id,
        "student_target_latent_id": target_latent_id,
        "split": str(source.get("split", "train")),
    }


def label_for_latent_id(latent_id: int, label_to_id: Mapping[str, int]) -> str:
    for label, label_id in label_to_id.items():
        if int(label_id) == latent_id:
            return label
    raise ValueError(f"No emotion label configured for latent id {latent_id}")


def strip_latent_markers(text: str, *, marker_template: str) -> str:
    stripped = text.strip()
    for latent_id in range(1000):
        marker = format_latent_marker(marker_template, latent_id)
        if stripped.endswith(marker):
            return stripped[: -len(marker)].strip()
    return stripped


def normalize_teacher_record(
    row: Mapping[str, Any],
    config: ExperimentConfig,
) -> DistillationRecord | None:
    output = str(row["teacher_output_text"]).strip()
    if not output:
        return None
    output = output[: config.distillation.max_teacher_output_chars].strip()
    output = strip_latent_markers(
        output,
        marker_template=config.soft_prompt.latent_marker_template,
    )
    if not output:
        return None

    target_latent_id = int(row["student_target_latent_id"])

    return DistillationRecord(
        base_input_text=str(row["base_input_text"]).strip(),
        teacher_instruction=str(row["teacher_instruction"]).strip(),
        teacher_output_text=output,
        emotion_label=str(row["emotion_label"]),
        student_input_latent_id=int(row["student_input_latent_id"]),
        student_target_latent_id=target_latent_id,
        split=str(row.get("split", "train")),
    )


def append_latent_marker(text: str, *, latent_id: int, marker_template: str) -> str:
    marker = format_latent_marker(marker_template, latent_id)
    stripped = strip_latent_markers(text, marker_template=marker_template)
    return f"{stripped}\n{marker}" if stripped else marker


def teacher_cache_key(request: Mapping[str, Any]) -> str:
    payload = {
        key: request[key]
        for key in (
            "base_input_text",
            "teacher_instruction",
            "emotion_label",
            "student_input_latent_id",
            "student_target_latent_id",
            "split",
        )
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generate_in_batches(
    generator: TeacherGenerator,
    prompts: Sequence[str],
    *,
    batch_size: int,
) -> Iterable[str]:
    for start in range(0, len(prompts), batch_size):
        yield from generator(prompts[start : start + batch_size])


class HuggingFaceTeacherGenerator:
    def __init__(self, config: ExperimentConfig) -> None:
        tokenizer_id = config.distillation.teacher_tokenizer or config.distillation.teacher_model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=config.distillation.teacher_trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        kwargs: dict[str, Any] = {
            "trust_remote_code": config.distillation.teacher_trust_remote_code,
        }
        if config.distillation.teacher_device_map is not None:
            kwargs["device_map"] = config.distillation.teacher_device_map
        dtype = resolve_torch_dtype(config.distillation.teacher_torch_dtype)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            config.distillation.teacher_model,
            **kwargs,
        )
        self.config = config

    @torch.no_grad()
    def __call__(self, prompts: Sequence[str]) -> Sequence[str]:
        encoded = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output_ids = self.model.generate(
            **encoded,
            **self._generation_kwargs(),
        )
        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        return [remove_prompt_prefix(text, prompt) for text, prompt in zip(decoded, prompts)]

    def _generation_kwargs(self) -> dict[str, Any]:
        do_sample = self.config.distillation.temperature > 0
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.distillation.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            kwargs["temperature"] = self.config.distillation.temperature
            kwargs["top_p"] = self.config.distillation.top_p
        return kwargs


def remove_prompt_prefix(generated_text: str, prompt: str) -> str:
    text = generated_text.strip()
    return text[len(prompt) :].strip() if text.startswith(prompt) else text


def resolve_teacher_cache_path(config: ExperimentConfig) -> Path:
    return config.distillation.teacher_cache_path or (
        config.output.run_dir / "teacher_generations.jsonl"
    )


def resolve_distill_data_path(config: ExperimentConfig) -> Path:
    return config.distillation.distill_data_path or (
        config.output.run_dir / "distillation_data.jsonl"
    )


def load_teacher_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    cache: dict[str, dict[str, Any]] = {}
    for row in rows:
        cache[teacher_cache_key(row)] = dict(row)
    return cache


def write_teacher_cache(records: Iterable[Mapping[str, Any]], path: Path) -> Path:
    return write_jsonl(records, path)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(records: Iterable[Mapping[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            file.write("\n")
    return path
