from __future__ import annotations

from llm_emotion_test.agents.negotiation import (
    RuleBasedConstraintSharingAgent,
    parse_agent_action,
)
from llm_emotion_test.config import ExperimentConfig, RLTaskConfig
from llm_emotion_test.tasks.hidden_constraints import (
    filter_codes,
    generate_hidden_constraint_problem,
)
from llm_emotion_test.tasks.negotiation_env import CooperativeHiddenConstraintsEnv


def test_hidden_constraint_problem_is_unique_but_private_views_are_not() -> None:
    config = RLTaskConfig(max_generation_attempts=500)

    problem = generate_hidden_constraint_problem(config, seed=123)

    all_codes = [
        tuple(int(char) for char in f"{number:04d}")
        for number in range(10_000)
        if len(set(f"{number:04d}")) == 4
    ]
    full_candidates = filter_codes(all_codes, problem.all_constraints)
    a_candidates = filter_codes(all_codes, problem.agent_constraints["A"])
    b_candidates = filter_codes(all_codes, problem.agent_constraints["B"])

    assert full_candidates == [problem.code]
    assert len(a_candidates) >= 2
    assert len(b_candidates) >= 2


def test_rule_based_agents_complete_episode() -> None:
    config = ExperimentConfig(
        rl_task=RLTaskConfig(max_generation_attempts=500, num_episodes=1, max_turns=3),
        training={"report_to": "none"},
    )
    env = CooperativeHiddenConstraintsEnv(config)
    agents = {
        "A": RuleBasedConstraintSharingAgent(),
        "B": RuleBasedConstraintSharingAgent(),
    }

    env.reset(seed=7)
    while not env.is_done:
        agent_id = env._require_state().active_agent_id
        env.step(agents[agent_id].act(env.observe(agent_id)))

    transcript = env.transcript_record()
    assert transcript["success"] is True
    assert transcript["total_reward"] > 1.0
    assert transcript["turn_evaluations"][-1]["provisional_answer"] == transcript["task"]["answer"]


def test_parse_agent_action_extracts_answer_and_latent() -> None:
    action = parse_agent_action(
        "候補はこれです。<answer>1234</answer><|emotion|>002<|/emotion|>",
        previous_latent_id=0,
        num_latents=8,
        marker_template="<|emotion|>{latent_id:03d}<|/emotion|>",
    )

    assert action.proposal == "1234"
    assert action.next_latent_id == 2
    assert "<answer>" not in action.message_text
