from __future__ import annotations

from types import SimpleNamespace

import torch

from llm_emotion_test.config import ExperimentConfig
from llm_emotion_test.models import loader


class DummyModel:
    config = SimpleNamespace(hidden_size=8)


def test_load_base_model_can_suppress_configured_device_map(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_from_pretrained(*_args, **kwargs):
        calls.append(kwargs)
        return DummyModel()

    monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
    config = ExperimentConfig.model_validate(
        {
            "model": {
                "base_model": "dummy/model",
                "device_map": "auto",
            },
        }
    )

    loader.load_base_model(config, device_map=None)

    assert "device_map" not in calls[0]


def test_load_base_model_uses_dtype_kwarg(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_from_pretrained(*_args, **kwargs):
        calls.append(kwargs)
        return DummyModel()

    monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
    config = ExperimentConfig.model_validate(
        {
            "model": {
                "base_model": "dummy/model",
                "device_map": None,
                "torch_dtype": "bfloat16",
            },
        }
    )

    loader.load_base_model(config)

    assert calls[0]["dtype"] is torch.bfloat16
    assert "torch_dtype" not in calls[0]
