from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from string import Formatter


class InvalidLatentFallback(StrEnum):
    PREVIOUS = "previous"
    NEUTRAL = "neutral"
    ERROR = "error"


@dataclass(frozen=True)
class LatentMarkerSpec:
    template: str = "<|emotion|>{latent_id:03d}<|/emotion|>"

    @property
    def start_token(self) -> str:
        return _template_literal_parts(self.template)[0]

    @property
    def end_token(self) -> str:
        return _template_literal_parts(self.template)[1]

    @property
    def special_tokens(self) -> list[str]:
        return [self.start_token, self.end_token]

    def format(self, latent_id: int) -> str:
        return self.template.format(latent_id=latent_id)

    def pattern(self) -> re.Pattern[str]:
        start, end = map(re.escape, _template_literal_parts(self.template))
        return re.compile(rf"{start}\s*(?P<latent_id>\d+)\s*{end}")


def _template_literal_parts(template: str) -> tuple[str, str]:
    before = ""
    after = ""
    seen_latent_id = False
    for literal_text, field_name, _format_spec, _conversion in Formatter().parse(template):
        if field_name is None:
            if seen_latent_id:
                after += literal_text
            else:
                before += literal_text
            continue
        if field_name != "latent_id":
            raise ValueError(f"Unsupported latent marker field: {field_name}")
        if seen_latent_id:
            raise ValueError("latent_marker_template can contain latent_id only once")
        before += literal_text
        seen_latent_id = True

    if not seen_latent_id:
        raise ValueError("latent_marker_template must include latent_id")
    if not before or not after:
        raise ValueError("latent_marker_template must wrap latent_id with literal tokens")
    return before, after


def parse_latent_id(
    text: str,
    *,
    num_latents: int,
    marker_template: str = "<|emotion|>{latent_id:03d}<|/emotion|>",
    previous_latent_id: int | None = None,
    fallback: str | InvalidLatentFallback = InvalidLatentFallback.PREVIOUS,
    neutral_latent_id: int = 0,
) -> int:
    """Extract the last valid latent marker from generated text."""

    spec = LatentMarkerSpec(marker_template)
    matches = list(spec.pattern().finditer(text))
    if matches:
        latent_id = int(matches[-1].group("latent_id"))
        if 0 <= latent_id < num_latents:
            return latent_id

    strategy = InvalidLatentFallback(fallback)
    if strategy is InvalidLatentFallback.PREVIOUS:
        if previous_latent_id is None:
            raise ValueError("previous_latent_id is required when fallback='previous'")
        return previous_latent_id
    if strategy is InvalidLatentFallback.NEUTRAL:
        if not 0 <= neutral_latent_id < num_latents:
            raise ValueError("neutral_latent_id must be within [0, num_latents)")
        return neutral_latent_id
    raise ValueError("Generated text does not contain a valid latent marker")


def add_latent_special_tokens(tokenizer, marker_template: str) -> int:
    spec = LatentMarkerSpec(marker_template)
    return tokenizer.add_special_tokens({"additional_special_tokens": spec.special_tokens})
