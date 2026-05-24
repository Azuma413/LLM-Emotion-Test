from pathlib import Path

import pytest

from llm_emotion_test.config import ConfigError, load_config


def test_load_default_sft_config() -> None:
    config = load_config(Path("configs/sft.yaml"))

    assert config.stage == "sft"
    assert config.output.run_dir == Path("outputs/runs/sft")
    assert config.output.checkpoints_dir == Path("outputs/runs/sft/checkpoints")
    assert config.training.max_steps is None


def test_config_validation_error_is_readable(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("soft_prompt:\n  prompt_length: 0\n", encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        load_config(config_path)

    message = str(exc_info.value)
    assert "Configuration validation failed" in message
    assert "soft_prompt.prompt_length" in message
