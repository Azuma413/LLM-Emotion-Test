from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
from datasets import load_dataset

from llm_emotion_test.config import ExperimentConfig


WRIME_EMOTION_LABELS = (
    "joy",
    "sadness",
    "anticipation",
    "surprise",
    "anger",
    "fear",
    "disgust",
    "trust",
)

WRIME_TSV_URLS = {
    "ver1": "https://raw.githubusercontent.com/ids-cv/wrime/master/wrime-ver1.tsv",
    "ver2": "https://raw.githubusercontent.com/ids-cv/wrime/master/wrime-ver2.tsv",
}


@dataclass(frozen=True)
class PreparedSample:
    input_text: str
    target_text: str
    emotion_labels: dict[str, int]
    input_latent_id: int
    target_latent_id: int
    split: str

    def as_json(self) -> dict[str, Any]:
        return {
            "input_text": self.input_text,
            "target_text": self.target_text,
            "emotion_labels": self.emotion_labels,
            "input_latent_id": self.input_latent_id,
            "target_latent_id": self.target_latent_id,
            "split": self.split,
        }


def load_wrime_dataset(config: ExperimentConfig) -> Mapping[str, Iterable[Mapping[str, Any]]]:
    if config.data.dataset_name == "shunk031/wrime":
        return load_wrime_tsv_dataset(config.data.dataset_config or "ver1")

    kwargs: dict[str, Any] = {}
    if config.data.dataset_config is not None:
        kwargs["name"] = config.data.dataset_config
    return load_dataset(config.data.dataset_name, **kwargs)


def load_wrime_tsv_dataset(version: str) -> dict[str, list[dict[str, Any]]]:
    if version not in WRIME_TSV_URLS:
        raise ValueError(f"Unsupported WRIME dataset_config: {version}")

    frame = pd.read_csv(WRIME_TSV_URLS[version], delimiter="\t")
    frame.columns = frame.columns.str.lower().str.replace(". ", "_", regex=False)
    split_names = {"train": "train", "dev": "validation", "test": "test"}
    dataset: dict[str, list[dict[str, Any]]] = {split: [] for split in split_names.values()}

    for _, row in frame.iterrows():
        split = split_names.get(str(row["train/dev/test"]))
        if split is None:
            continue
        dataset[split].append(convert_wrime_tsv_row(row))

    return dataset


def convert_wrime_tsv_row(row: pd.Series) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sentence": row["sentence"],
        "user_id": str(row["userid"]),
        "datetime": row["datetime"],
    }
    for source in ("writer", "reader1", "reader2", "reader3", "avg_readers"):
        record[source] = {
            label: int(row.get(f"{source}_{label}", 0)) for label in WRIME_EMOTION_LABELS
        }
    return record


def prepare_wrime_dataset(config: ExperimentConfig) -> dict[str, Any]:
    label_to_id = build_label_to_id(config.data.emotion_labels, config.data.representative_label_map)
    if config.soft_prompt.num_latents != len(label_to_id):
        raise ValueError(
            "soft_prompt.num_latents must match the normalized emotion label count "
            f"({config.soft_prompt.num_latents} != {len(label_to_id)})"
        )

    dataset = load_wrime_dataset(config)
    rng = random.Random(config.data.seed)
    samples = iter_prepared_samples(
        dataset=dataset,
        text_column=config.data.text_column,
        annotation_source=config.data.annotation_source,
        label_to_id=label_to_id,
        representative_label_map=config.data.representative_label_map,
        latent_marker_template=config.soft_prompt.latent_marker_template,
        copy_input_latent_probability=config.data.copy_input_latent_probability,
        rng=rng,
        max_samples=config.data.max_samples,
    )
    records = [sample.as_json() for sample in samples]
    output_path = write_jsonl(records, config.data.processed_path)
    stats = summarize_records(records)
    stats["output_path"] = str(output_path)
    stats["label_to_id"] = label_to_id
    return stats


def build_label_to_id(
    emotion_labels: Iterable[str], representative_label_map: Mapping[str, str] | None = None
) -> dict[str, int]:
    representative_label_map = representative_label_map or {}
    normalized_labels: list[str] = []
    for label in emotion_labels:
        normalized = representative_label_map.get(label, label)
        if normalized not in normalized_labels:
            normalized_labels.append(normalized)
    return {label: index for index, label in enumerate(normalized_labels)}


def iter_prepared_samples(
    dataset: Mapping[str, Iterable[Mapping[str, Any]]],
    text_column: str,
    annotation_source: str,
    label_to_id: Mapping[str, int],
    representative_label_map: Mapping[str, str],
    latent_marker_template: str,
    copy_input_latent_probability: float,
    rng: random.Random,
    max_samples: int | None = None,
) -> Iterable[PreparedSample]:
    count = 0
    for split, rows in dataset.items():
        for row in rows:
            if max_samples is not None and count >= max_samples:
                return
            yield convert_wrime_row(
                row=row,
                split=split,
                text_column=text_column,
                annotation_source=annotation_source,
                label_to_id=label_to_id,
                representative_label_map=representative_label_map,
                latent_marker_template=latent_marker_template,
                copy_input_latent_probability=copy_input_latent_probability,
                rng=rng,
            )
            count += 1


def convert_wrime_row(
    row: Mapping[str, Any],
    split: str,
    text_column: str,
    annotation_source: str,
    label_to_id: Mapping[str, int],
    representative_label_map: Mapping[str, str],
    latent_marker_template: str,
    copy_input_latent_probability: float,
    rng: random.Random,
) -> PreparedSample:
    text = str(row[text_column]).strip()
    emotion_labels = normalize_emotion_labels(
        row[annotation_source], label_to_id, representative_label_map
    )
    primary_label = max(
        label_to_id,
        key=lambda label: (emotion_labels.get(label, 0), -label_to_id[label]),
    )
    input_latent_id = label_to_id[primary_label]
    if rng.random() < copy_input_latent_probability:
        target_latent_id = input_latent_id
    else:
        target_latent_id = rng.randrange(len(label_to_id))

    input_marker = format_latent_marker(latent_marker_template, input_latent_id)
    target_marker = format_latent_marker(latent_marker_template, target_latent_id)
    input_text = f"{text}\n{input_marker}"
    target_text = f"{text}\n{target_marker}"

    return PreparedSample(
        input_text=input_text,
        target_text=target_text,
        emotion_labels=emotion_labels,
        input_latent_id=input_latent_id,
        target_latent_id=target_latent_id,
        split=split,
    )


def normalize_emotion_labels(
    raw_labels: Mapping[str, Any],
    label_to_id: Mapping[str, int],
    representative_label_map: Mapping[str, str],
) -> dict[str, int]:
    normalized: defaultdict[str, int] = defaultdict(int)
    for source_label in WRIME_EMOTION_LABELS:
        value = raw_labels.get(source_label, 0)
        target_label = representative_label_map.get(source_label, source_label)
        if target_label in label_to_id:
            normalized[target_label] += int(value)
    return {label: normalized[label] for label in label_to_id}


def format_latent_marker(template: str, latent_id: int) -> str:
    return template.format(latent_id=latent_id)


def write_jsonl(records: Iterable[Mapping[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            file.write("\n")
    return output_path


def summarize_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    text_lengths = [len(str(row["target_text"])) for row in rows]
    split_counts = Counter(str(row["split"]) for row in rows)
    input_latent_counts = Counter(str(row["input_latent_id"]) for row in rows)
    target_latent_counts = Counter(str(row["target_latent_id"]) for row in rows)
    label_counts: Counter[str] = Counter()
    for row in rows:
        labels = row["emotion_labels"]
        if labels:
            primary = max(labels, key=lambda label: labels[label])
            label_counts[primary] += 1

    return {
        "num_samples": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "input_latent_counts": dict(sorted(input_latent_counts.items())),
        "target_latent_counts": dict(sorted(target_latent_counts.items())),
        "text_length": summarize_numbers(text_lengths),
    }


def summarize_numbers(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(values), "max": max(values), "mean": mean(values)}
