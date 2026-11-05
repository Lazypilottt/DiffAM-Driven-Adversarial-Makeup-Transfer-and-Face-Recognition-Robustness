import pytest

from makeup_attack.budget import (
    estimate_budget,
    fooling_probability,
    format_table,
    similarity_tier,
    similarity_to_steps,
)


def test_steps_hit_bounds():
    assert similarity_to_steps(1.0) == 5
    assert similarity_to_steps(0.0) == 40
    assert similarity_to_steps(1.7) == 5
    assert similarity_to_steps(-0.3) == 40


def test_steps_decrease_with_similarity():
    sims = [i / 20 for i in range(21)]
    steps = [similarity_to_steps(s) for s in sims]
    assert steps == sorted(steps, reverse=True)


def test_steps_rejects_inverted_range():
    with pytest.raises(ValueError):
        similarity_to_steps(0.5, min_steps=10, max_steps=5)


def test_fooling_probability_is_half_at_midpoint():
    assert fooling_probability(0.6) == pytest.approx(0.5)
    assert fooling_probability(0.8, midpoint=0.8) == pytest.approx(0.5)


def test_fooling_probability_increases_with_similarity():
    assert fooling_probability(0.3) < fooling_probability(0.6) < fooling_probability(0.95)
    assert 0.0 < fooling_probability(0.0) < fooling_probability(1.0) < 1.0


@pytest.mark.parametrize("sim, tier", [(0.95, "high"), (0.80, "high"), (0.70, "medium"),
                                       (0.65, "medium"), (0.64, "low"), (0.1, "low")])
def test_tiers(sim, tier):
    assert similarity_tier(sim) == tier


def test_estimate_budget_bundles_fields():
    b = estimate_budget(0.92)
    assert b.tier == "high"
    assert b.steps == similarity_to_steps(0.92)
    assert b.fooling_probability == pytest.approx(fooling_probability(0.92))


def test_format_table_has_one_row_per_score():
    table = format_table([0.9, 0.5, 0.2])
    assert len(table.splitlines()) == 2 + 3
