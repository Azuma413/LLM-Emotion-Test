"""Model wrappers, soft prompts, and checkpoint utilities."""
from llm_emotion_test.models.latent import (
    InvalidLatentFallback,
    LatentMarkerSpec,
    add_latent_special_tokens,
    parse_latent_id,
)
from llm_emotion_test.models.loader import (
    build_soft_prompt_model,
    load_soft_prompt_model_from_checkpoint,
    load_tokenizer,
    save_model_checkpoint,
)
from llm_emotion_test.models.soft_prompt import (
    SoftPromptCausalLM,
    SoftPromptEmbedding,
    load_soft_prompt,
)

__all__ = [
    "InvalidLatentFallback",
    "LatentMarkerSpec",
    "SoftPromptCausalLM",
    "SoftPromptEmbedding",
    "add_latent_special_tokens",
    "build_soft_prompt_model",
    "load_soft_prompt",
    "load_soft_prompt_model_from_checkpoint",
    "load_tokenizer",
    "parse_latent_id",
    "save_model_checkpoint",
]
