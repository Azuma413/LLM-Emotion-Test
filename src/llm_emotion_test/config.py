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


class DistillationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    teacher_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    teacher_tokenizer: str | None = None
    teacher_trust_remote_code: bool = True
    teacher_torch_dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    teacher_device_map: str | dict[str, Any] | None = "auto"
    teacher_batch_size: int = Field(default=1, ge=1)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_new_tokens: int = Field(default=128, ge=1)
    source_data_path: Path = Path("datasets/processed/sft-smoke.jsonl")
    teacher_cache_path: Path | None = None
    distill_data_path: Path | None = None
    overwrite_cache: bool = False
    require_teacher_latent_marker: bool = False
    max_teacher_output_chars: int = Field(default=1000, ge=1)
    deduplicate: bool = True
    kl_divergence_weight: float = Field(default=0.0, ge=0.0)


class RLTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_backend: Literal["llm", "tabular_smoke"] = "llm"
    code_length: int = Field(default=4, ge=4, le=5)
    digits: list[int] = Field(default_factory=lambda: list(range(10)))
    allow_repeats: bool = False
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    num_agents: int = Field(default=2, ge=2, le=2)
    min_constraints_per_agent: int = Field(default=2, ge=1)
    max_constraints_per_agent: int = Field(default=4, ge=1)
    min_private_candidates: int = Field(default=2, ge=2)
    max_candidate_balance_ratio: float = Field(default=4.0, ge=1.0)
    max_generation_attempts: int = Field(default=200, ge=1)
    max_turns: int = Field(default=3, ge=1)
    num_episodes: int = Field(default=2, ge=1)
    rollouts_per_problem: int = Field(default=4, ge=1)
    reward_team_weight: float = Field(default=1.0, ge=0.0)
    latent_policy_learning_rate: float = Field(default=0.1, gt=0.0)
    train_base_model: bool = False
    sampling_temperature: float = Field(default=1.0, ge=0.0)
    sampling_top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    grpo_advantage_epsilon: float = Field(default=1e-6, gt=0.0)
    eval_episodes: int = Field(default=2, ge=1)
    checkpoint_filename: str = "rl_checkpoint.json"
    resume_from_checkpoint: Path | None = None
    transcript_filename: str = "rl_transcripts.jsonl"
    allowed_constraint_types: list[str] = Field(
        default_factory=lambda: [
            "parity",
            "comparison",
            "difference",
            "sum",
            "forbidden_values",
            "allowed_values",
            "all_distinct",
            "contains",
            "not_contains",
            "parity_count",
        ]
    )

    @field_validator("digits")
    @classmethod
    def digits_are_unique_decimal_values(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("digits must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("digits must be unique")
        if any(digit < 0 or digit > 9 for digit in value):
            raise ValueError("digits must be decimal values in [0, 9]")
        return value

    @field_validator("max_constraints_per_agent")
    @classmethod
    def max_constraints_not_less_than_min(cls, value: int, info) -> int:
        min_value = info.data.get("min_constraints_per_agent")
        if min_value is not None and value < min_value:
            raise ValueError(
                "max_constraints_per_agent must be >= min_constraints_per_agent"
            )
        return value


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_tasks: int = Field(default=8, ge=1)
    task_seed_offset: int = 50_000
    variants: list[
        Literal[
            "base_model",
            "sft_model",
            "distilled_model",
            "rl_model",
            "latent_fixed",
            "latent_random",
            "no_latent",
        ]
    ] = Field(
        default_factory=lambda: [
            "base_model",
            "sft_model",
            "distilled_model",
            "rl_model",
            "latent_fixed",
            "latent_random",
            "no_latent",
        ]
    )
    source_run_dir: Path | None = None
    transcript_sample_count: int = Field(default=3, ge=0)
    failure_sample_count: int = Field(default=3, ge=0)
    metrics_csv_filename: str = "evaluation_metrics.csv"
    report_filename: str = "evaluation_report.md"
    transcript_filename: str = "evaluation_transcripts.jsonl"
    comparison_filename: str = "model_comparison.jsonl"
    latent_heatmap_filename: str = "latent_transition_heatmap.svg"
    reward_curve_filename: str = "reward_curve.svg"
    emotion_distribution_filename: str = "emotion_distribution.svg"


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
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)
    rl_task: RLTaskConfig = Field(default_factory=RLTaskConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
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
