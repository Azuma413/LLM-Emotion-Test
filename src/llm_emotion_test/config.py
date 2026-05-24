from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ConfigError(ValueError):
    """Raised when a configuration file cannot be loaded or validated."""


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer: str | None = None
    trust_remote_code: bool = True
    torch_dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    device_map: str | dict[str, Any] | None = "auto"
    use_lora: bool = False
    use_qlora: bool = False
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    lora_target_modules: list[str] | None = None


class SoftPromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_latents: int = Field(default=8, ge=1)
    prompt_length: int = Field(default=4, ge=1)
    init_strategy: Literal["normal", "mean_token", "zeros"] = "normal"
    latent_marker_template: str = "<|emotion|>{latent_id:03d}<|/emotion|>"
    invalid_latent_fallback: Literal["previous", "neutral", "error"] = "previous"
    neutral_latent_id: int = Field(default=0, ge=0)

    @field_validator("latent_marker_template")
    @classmethod
    def latent_marker_template_has_id(cls, value: str) -> str:
        if "{latent_id" not in value:
            raise ValueError("latent_marker_template must include a {latent_id} placeholder")
        return value

    @field_validator("neutral_latent_id")
    @classmethod
    def neutral_latent_id_is_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("neutral_latent_id must be non-negative")
        return value


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str = "shunk031/wrime"
    dataset_config: str | None = "ver1"
    raw_dir: Path = Path("datasets/raw")
    processed_dir: Path = Path("datasets/processed")
    processed_filename: str = "sft.jsonl"
    text_column: str = "sentence"
    annotation_source: Literal["writer", "reader1", "reader2", "reader3", "avg_readers"] = (
        "writer"
    )
    emotion_labels: list[str] = Field(
        default_factory=lambda: [
            "joy",
            "sadness",
            "anticipation",
            "surprise",
            "anger",
            "fear",
            "disgust",
            "trust",
        ]
    )
    representative_label_map: dict[str, str] = Field(default_factory=dict)
    copy_input_latent_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    max_samples: int | None = Field(default=None, ge=1)
    seed: int = 42

    @property
    def processed_path(self) -> Path:
        return self.processed_dir / self.processed_filename


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(default=2e-5, gt=0)
    batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_steps: int = Field(default=10, ge=1)
    max_seq_length: int = Field(default=512, ge=8)
    precision: Literal["fp32", "fp16", "bf16"] = "bf16"
    logging_steps: int = Field(default=1, ge=1)
    eval_steps: int | None = Field(default=None, ge=1)
    save_steps: int | None = Field(default=None, ge=1)
    warmup_steps: int = Field(default=0, ge=0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    num_train_epochs: float = Field(default=1.0, gt=0.0)
    eval_max_samples: int = Field(default=16, ge=1)
    sample_count: int = Field(default=4, ge=0)
    generation_max_new_tokens: int = Field(default=64, ge=1)
    report_to: Literal["wandb", "tensorboard", "none"] = "wandb"


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: Path = Path("outputs/runs")
    run_id: str = "smoke"

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_id

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.yaml"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.jsonl"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def samples_path(self) -> Path:
        return self.run_dir / "samples.jsonl"


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str = "auto"
    seed: int = 42


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: str = "llm-emotion-test"
    stage: Literal["base", "sft", "distill", "rl_grpo", "eval"] = "base"
    model: ModelConfig = Field(default_factory=ModelConfig)
    soft_prompt: SoftPromptConfig = Field(default_factory=SoftPromptConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Could not read config file: {config_path}") from exc

    try:
        data = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in config file: {config_path}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Config root must be a YAML mapping/object")

    try:
        return ExperimentConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc)) from exc


def format_validation_error(error: ValidationError) -> str:
    lines = ["Configuration validation failed:"]
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"])
        message = item["msg"]
        lines.append(f"- {location}: {message}")
    return "\n".join(lines)


def config_summary(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "stage": config.stage,
        "base_model": config.model.base_model,
        "dataset": config.data.dataset_name,
        "run_dir": str(config.output.run_dir),
        "report_to": config.training.report_to,
    }
