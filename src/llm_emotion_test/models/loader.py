from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm_emotion_test.config import ExperimentConfig
from llm_emotion_test.data.wrime import build_label_to_id
from llm_emotion_test.models.latent import add_latent_special_tokens
from llm_emotion_test.models.soft_prompt import SoftPromptCausalLM, SoftPromptEmbedding, load_soft_prompt


def load_tokenizer(config: ExperimentConfig):
    tokenizer_id = config.model.tokenizer or config.model.base_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=config.model.trust_remote_code,
    )
    add_latent_special_tokens(tokenizer, config.soft_prompt.latent_marker_template)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(config: ExperimentConfig, *, apply_adapters: bool = True):
    kwargs: dict[str, Any] = {
        "trust_remote_code": config.model.trust_remote_code,
    }
    if config.model.device_map is not None:
        kwargs["device_map"] = config.model.device_map
    dtype = resolve_torch_dtype(config.model.torch_dtype)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if config.model.use_qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype or torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(config.model.base_model, **kwargs)
    if config.model.use_qlora:
        model = prepare_model_for_kbit_training(model)
    if apply_adapters and (config.model.use_lora or config.model.use_qlora):
        model = apply_lora(model, config)
    return model


def build_soft_prompt_model(config: ExperimentConfig) -> tuple[SoftPromptCausalLM, Any]:
    require_cuda_if_configured(config)
    tokenizer = load_tokenizer(config)
    base_model = load_base_model(config)
    base_model.resize_token_embeddings(len(tokenizer))

    hidden_size = get_hidden_size(base_model)
    token_embedding = base_model.get_input_embeddings()
    soft_prompt = SoftPromptEmbedding(
        num_latents=config.soft_prompt.num_latents,
        prompt_length=config.soft_prompt.prompt_length,
        hidden_size=hidden_size,
        init_strategy=config.soft_prompt.init_strategy,
        token_embedding=token_embedding,
    )
    return SoftPromptCausalLM(base_model, soft_prompt), tokenizer


def load_soft_prompt_model_from_checkpoint(
    checkpoint_dir: str | Path,
    config: ExperimentConfig,
) -> tuple[SoftPromptCausalLM, Any]:
    require_cuda_if_configured(config)
    checkpoint_path = Path(checkpoint_dir)
    tokenizer = load_tokenizer(config)
    saved_model_path = checkpoint_path / "base_or_adapter"
    if (saved_model_path / "adapter_config.json").exists():
        base_model = load_base_model(config, apply_adapters=False)
        base_model.resize_token_embeddings(len(tokenizer))
        base_model = PeftModel.from_pretrained(base_model, saved_model_path)
    elif saved_model_path.exists():
        base_model = load_saved_base_model(saved_model_path, config)
    else:
        base_model = load_base_model(config)
    base_model.resize_token_embeddings(len(tokenizer))
    soft_prompt = load_soft_prompt(checkpoint_path / "soft_prompt.pt")
    return SoftPromptCausalLM(base_model, soft_prompt), tokenizer


def load_saved_base_model(model_dir: str | Path, config: ExperimentConfig):
    kwargs: dict[str, Any] = {
        "trust_remote_code": config.model.trust_remote_code,
    }
    if config.model.device_map is not None:
        kwargs["device_map"] = config.model.device_map
    dtype = resolve_torch_dtype(config.model.torch_dtype)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    return AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)


def resolve_torch_dtype(value: str) -> torch.dtype | None:
    if value == "auto":
        return None
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {value}")


def require_cuda_if_configured(config: ExperimentConfig) -> None:
    if config.runtime.require_gpu and not torch.cuda.is_available():
        raise RuntimeError("This run requires a CUDA GPU, but torch.cuda.is_available() is false")


def get_hidden_size(model) -> int:
    config = model.config
    for attr in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    embedding = model.get_input_embeddings()
    return int(embedding.embedding_dim)


def apply_lora(model, config: ExperimentConfig):
    lora_config = LoraConfig(
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        target_modules=config.model.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def save_model_checkpoint(
    model: SoftPromptCausalLM,
    tokenizer,
    config: ExperimentConfig,
    checkpoint_dir: str | Path,
) -> Path:
    label_mapping = build_label_to_id(
        config.data.emotion_labels,
        config.data.representative_label_map,
    )
    return model.save_checkpoint(
        checkpoint_dir,
        base_model_id=config.model.base_model,
        tokenizer=tokenizer,
        emotion_label_mapping=label_mapping,
    )
