import random

from llm_emotion_test.data.wrime import (
    build_label_to_id,
    convert_wrime_row,
    iter_prepared_samples,
    summarize_records,
)


def test_convert_wrime_row_uses_primary_emotion_as_input_latent() -> None:
    label_to_id = build_label_to_id(["joy", "sadness", "anger"])
    row = {
        "sentence": "今日は少しつらい。",
        "writer": {
            "joy": 0,
            "sadness": 3,
            "anticipation": 0,
            "surprise": 0,
            "anger": 1,
            "fear": 0,
            "disgust": 0,
            "trust": 0,
        },
    }

    sample = convert_wrime_row(
        row=row,
        split="train",
        text_column="sentence",
        annotation_source="writer",
        label_to_id=label_to_id,
        representative_label_map={},
        latent_marker_template="<|emotion|>{latent_id:03d}<|/emotion|>",
        copy_input_latent_probability=1.0,
        rng=random.Random(0),
    )

    assert sample.input_latent_id == label_to_id["sadness"]
    assert sample.target_latent_id == sample.input_latent_id
    assert sample.emotion_labels == {"joy": 0, "sadness": 3, "anger": 1}
    assert sample.input_text.endswith("<|emotion|>001<|/emotion|>")
    assert sample.target_text.endswith("<|emotion|>001<|/emotion|>")


def test_iter_prepared_samples_respects_global_max_samples() -> None:
    rows = {
        "train": [
            {"sentence": "a", "writer": {"joy": 1}},
            {"sentence": "b", "writer": {"sadness": 1}},
        ],
        "validation": [{"sentence": "c", "writer": {"anger": 1}}],
    }

    samples = list(
        iter_prepared_samples(
            dataset=rows,
            text_column="sentence",
            annotation_source="writer",
            label_to_id=build_label_to_id(["joy", "sadness", "anger"]),
            representative_label_map={},
            latent_marker_template="<|emotion|>{latent_id:03d}<|/emotion|>",
            copy_input_latent_probability=1.0,
            rng=random.Random(0),
            max_samples=2,
        )
    )

    assert [sample.split for sample in samples] == ["train", "train"]


def test_summarize_records_counts_distribution() -> None:
    records = [
        {
            "input_text": "a",
            "target_text": "aa",
            "emotion_labels": {"joy": 2, "sadness": 0},
            "input_latent_id": 0,
            "target_latent_id": 0,
            "split": "train",
        },
        {
            "input_text": "b",
            "target_text": "bbb",
            "emotion_labels": {"joy": 0, "sadness": 3},
            "input_latent_id": 1,
            "target_latent_id": 0,
            "split": "validation",
        },
    ]

    summary = summarize_records(records)

    assert summary["num_samples"] == 2
    assert summary["split_counts"] == {"train": 1, "validation": 1}
    assert summary["label_counts"] == {"joy": 1, "sadness": 1}
    assert summary["input_latent_counts"] == {"0": 1, "1": 1}
    assert summary["target_latent_counts"] == {"0": 2}
    assert summary["text_length"]["min"] == 2
    assert summary["text_length"]["max"] == 3
