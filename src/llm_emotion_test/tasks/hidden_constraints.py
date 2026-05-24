from __future__ import annotations

import itertools
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from llm_emotion_test.config import RLTaskConfig


Code = tuple[int, ...]


@dataclass(frozen=True)
class Constraint:
    type: str
    params: dict[str, Any]
    text: str

    def is_satisfied(self, code: Sequence[int]) -> bool:
        if self.type == "parity":
            return code[self.params["position"]] % 2 == self.params["parity"]
        if self.type == "comparison":
            left = code[self.params["left_position"]]
            right = code[self.params["right_position"]]
            return left < right if self.params["operator"] == "<" else left > right
        if self.type == "difference":
            left = code[self.params["left_position"]]
            right = code[self.params["right_position"]]
            return left == right + self.params["delta"]
        if self.type == "sum":
            return sum(code) == self.params["value"]
        if self.type == "forbidden_values":
            return code[self.params["position"]] not in set(self.params["values"])
        if self.type == "allowed_values":
            return code[self.params["position"]] in set(self.params["values"])
        if self.type == "all_distinct":
            return len(set(code)) == len(code)
        if self.type == "contains":
            return self.params["value"] in code
        if self.type == "not_contains":
            return self.params["value"] not in code
        if self.type == "parity_count":
            parity = self.params["parity"]
            return sum(1 for digit in code if digit % 2 == parity) == self.params["count"]
        raise ValueError(f"Unsupported constraint type: {self.type}")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "params": dict(self.params), "text": self.text}


@dataclass(frozen=True)
class HiddenConstraintProblem:
    code: Code
    agent_constraints: dict[str, list[Constraint]]
    all_constraints: list[Constraint]
    metadata: dict[str, Any]

    @property
    def answer(self) -> str:
        return "".join(str(digit) for digit in self.code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "agent_constraints": {
                agent_id: [constraint.to_dict() for constraint in constraints]
                for agent_id, constraints in self.agent_constraints.items()
            },
            "all_constraints": [constraint.to_dict() for constraint in self.all_constraints],
            "metadata": dict(self.metadata),
        }


def generate_hidden_constraint_problem(
    config: RLTaskConfig,
    *,
    seed: int | None = None,
) -> HiddenConstraintProblem:
    rng = random.Random(seed)
    all_codes = enumerate_codes(
        code_length=config.code_length,
        digits=config.digits,
        allow_repeats=config.allow_repeats,
    )
    if not all_codes:
        raise ValueError("No candidate codes can be generated from RL task settings")

    for attempt in range(config.max_generation_attempts):
        code = rng.choice(all_codes)
        candidates = build_constraint_candidates(
            code,
            digits=config.digits,
            allowed_types=config.allowed_constraint_types,
            rng=rng,
        )
        rng.shuffle(candidates)
        selected = select_constraints_for_unique_solution(code, candidates, all_codes)
        if selected is None:
            continue
        selected = add_extra_true_constraints(
            selected,
            candidates,
            min_total=config.min_constraints_per_agent * 2,
            max_total=config.max_constraints_per_agent * 2,
        )
        selected = trim_redundant_constraints(
            code,
            selected,
            all_codes,
            min_total=config.min_constraints_per_agent * 2,
        )
        split = split_constraints_for_agents(config, selected, all_codes, rng)
        if split is None:
            continue
        agent_constraints, metadata = split
        metadata.update(
            {
                "attempt": attempt + 1,
                "code_length": config.code_length,
                "difficulty": config.difficulty,
                "all_candidate_count": len(all_codes),
                "full_candidate_count": len(
                    filter_codes(all_codes, selected)
                ),
                "constraint_types": sorted({constraint.type for constraint in selected}),
            }
        )
        return HiddenConstraintProblem(
            code=code,
            agent_constraints=agent_constraints,
            all_constraints=selected,
            metadata=metadata,
        )
    raise ValueError("Could not generate a valid hidden-constraint problem")


def enumerate_codes(
    *,
    code_length: int,
    digits: Sequence[int],
    allow_repeats: bool,
) -> list[Code]:
    iterator: Iterable[tuple[int, ...]]
    if allow_repeats:
        iterator = itertools.product(digits, repeat=code_length)
    else:
        iterator = itertools.permutations(digits, code_length)
    return [tuple(code) for code in iterator]


def filter_codes(codes: Sequence[Code], constraints: Sequence[Constraint]) -> list[Code]:
    return [code for code in codes if all(c.is_satisfied(code) for c in constraints)]


def build_constraint_candidates(
    code: Code,
    *,
    digits: Sequence[int],
    allowed_types: Sequence[str],
    rng: random.Random,
) -> list[Constraint]:
    allowed = set(allowed_types)
    constraints: list[Constraint] = []
    length = len(code)

    if "all_distinct" in allowed and len(set(code)) == length:
        constraints.append(Constraint("all_distinct", {}, "全桁は異なります。"))

    if "sum" in allowed:
        constraints.append(
            Constraint("sum", {"value": sum(code)}, f"全桁の合計は{sum(code)}です。")
        )

    if "parity_count" in allowed:
        even_count = sum(1 for digit in code if digit % 2 == 0)
        constraints.append(
            Constraint(
                "parity_count",
                {"parity": 0, "count": even_count},
                f"偶数の桁は{even_count}個です。",
            )
        )

    for position, digit in enumerate(code):
        human_pos = position + 1
        if "parity" in allowed:
            parity_text = "偶数" if digit % 2 == 0 else "奇数"
            constraints.append(
                Constraint(
                    "parity",
                    {"position": position, "parity": digit % 2},
                    f"{human_pos}桁目は{parity_text}です。",
                )
            )
        if "allowed_values" in allowed:
            values = sorted({digit, *rng.sample(list(digits), k=min(2, len(digits)))})
            constraints.append(
                Constraint(
                    "allowed_values",
                    {"position": position, "values": values},
                    f"{human_pos}桁目は{format_values(values)}のいずれかです。",
                )
            )
        if "forbidden_values" in allowed:
            forbidden_pool = [value for value in digits if value != digit]
            values = sorted(rng.sample(forbidden_pool, k=min(3, len(forbidden_pool))))
            constraints.append(
                Constraint(
                    "forbidden_values",
                    {"position": position, "values": values},
                    f"{human_pos}桁目は{format_values(values)}ではありません。",
                )
            )
        if "contains" in allowed:
            constraints.append(
                Constraint("contains", {"value": digit}, f"{digit}を含みます。")
            )

    if "not_contains" in allowed:
        for value in digits:
            if value not in code:
                constraints.append(
                    Constraint("not_contains", {"value": value}, f"{value}を含みません。")
                )

    for left in range(length):
        for right in range(length):
            if left == right:
                continue
            if "comparison" in allowed and code[left] != code[right]:
                operator = "<" if code[left] < code[right] else ">"
                constraints.append(
                    Constraint(
                        "comparison",
                        {
                            "left_position": left,
                            "right_position": right,
                            "operator": operator,
                        },
                        f"{left + 1}桁目は{right + 1}桁目より{'小さい' if operator == '<' else '大きい'}です。",
                    )
                )
            if "difference" in allowed:
                delta = code[left] - code[right]
                if delta in {-2, -1, 1, 2}:
                    constraints.append(
                        Constraint(
                            "difference",
                            {
                                "left_position": left,
                                "right_position": right,
                                "delta": delta,
                            },
                            difference_text(left, right, delta),
                        )
                    )

    return deduplicate_constraints(
        [constraint for constraint in constraints if constraint.is_satisfied(code)]
    )


def select_constraints_for_unique_solution(
    code: Code,
    candidates: Sequence[Constraint],
    all_codes: Sequence[Code],
) -> list[Constraint] | None:
    selected: list[Constraint] = []
    remaining = list(all_codes)
    for constraint in candidates:
        next_remaining = filter_codes(remaining, [constraint])
        if not next_remaining or code not in next_remaining:
            continue
        if len(next_remaining) < len(remaining):
            selected.append(constraint)
            remaining = next_remaining
        if len(remaining) == 1:
            return selected
    return None


def add_extra_true_constraints(
    selected: Sequence[Constraint],
    candidates: Sequence[Constraint],
    *,
    min_total: int,
    max_total: int,
) -> list[Constraint]:
    chosen = list(selected)
    chosen_keys = {constraint_key(constraint) for constraint in chosen}
    for candidate in candidates:
        if len(chosen) >= min_total:
            break
        key = constraint_key(candidate)
        if key in chosen_keys:
            continue
        chosen.append(candidate)
        chosen_keys.add(key)
    return chosen[:max_total]


def trim_redundant_constraints(
    code: Code,
    constraints: Sequence[Constraint],
    all_codes: Sequence[Code],
    *,
    min_total: int,
) -> list[Constraint]:
    selected = list(constraints)
    changed = True
    while changed:
        changed = False
        if len(selected) <= min_total:
            break
        for constraint in list(selected):
            trial = [item for item in selected if item is not constraint]
            if len(trial) < min_total:
                continue
            if filter_codes(all_codes, trial) == [code]:
                selected = trial
                changed = True
                break
    return selected


def split_constraints_for_agents(
    config: RLTaskConfig,
    constraints: Sequence[Constraint],
    all_codes: Sequence[Code],
    rng: random.Random,
) -> tuple[dict[str, list[Constraint]], dict[str, Any]] | None:
    if len(constraints) < config.min_constraints_per_agent * 2:
        constraints = list(constraints)
    for _ in range(200):
        shuffled = list(constraints)
        rng.shuffle(shuffled)
        split_index = len(shuffled) // 2
        agent_a = shuffled[:split_index]
        agent_b = shuffled[split_index:]
        if not (
            config.min_constraints_per_agent <= len(agent_a) <= config.max_constraints_per_agent
            and config.min_constraints_per_agent
            <= len(agent_b)
            <= config.max_constraints_per_agent
        ):
            continue
        a_candidates = filter_codes(all_codes, agent_a)
        b_candidates = filter_codes(all_codes, agent_b)
        full_candidates = filter_codes(all_codes, [*agent_a, *agent_b])
        if len(full_candidates) != 1:
            continue
        if len(a_candidates) < config.min_private_candidates:
            continue
        if len(b_candidates) < config.min_private_candidates:
            continue
        ratio = max(len(a_candidates), len(b_candidates)) / min(
            len(a_candidates), len(b_candidates)
        )
        if ratio > config.max_candidate_balance_ratio:
            continue
        if len({constraint.type for constraint in constraints}) < 2:
            continue
        return (
            {"A": agent_a, "B": agent_b},
            {
                "agent_candidate_counts": {
                    "A": len(a_candidates),
                    "B": len(b_candidates),
                },
                "candidate_balance_ratio": ratio,
            },
        )
    return None


def deduplicate_constraints(constraints: Sequence[Constraint]) -> list[Constraint]:
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    unique: list[Constraint] = []
    for constraint in constraints:
        key = constraint_key(constraint)
        if key in seen:
            continue
        seen.add(key)
        unique.append(constraint)
    return unique


def constraint_key(constraint: Constraint) -> tuple[str, tuple[tuple[str, str], ...]]:
    return (
        constraint.type,
        tuple(sorted((key, repr(value)) for key, value in constraint.params.items())),
    )


def format_values(values: Sequence[int]) -> str:
    return "、".join(str(value) for value in values)


def difference_text(left: int, right: int, delta: int) -> str:
    if delta > 0:
        return f"{left + 1}桁目は{right + 1}桁目より{delta}大きいです。"
    return f"{left + 1}桁目は{right + 1}桁目より{abs(delta)}小さいです。"
