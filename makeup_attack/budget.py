"""Turn a face-similarity score into a DiffAM attack budget.

Working hypothesis: a face that already sits close to another identity in
ArcFace embedding space needs a smaller push to be misidentified. Such faces
get a smaller perturbation budget and a higher predicted fooling probability.

The budget is expressed as the number of DDIM generative steps DiffAM takes
with the adversarially fine-tuned model (``--n_test_step``). Both mappings
below are hand-set heuristics, not curves fitted to measured attack results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

HIGH_SIMILARITY = 0.80
MEDIUM_SIMILARITY = 0.65


@dataclass(frozen=True)
class AttackBudget:
    similarity: float
    steps: int
    fooling_probability: float
    tier: str


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def similarity_to_steps(similarity: float, min_steps: int = 5, max_steps: int = 40) -> int:
    """Linear map from similarity in [0, 1] to a step budget.

    similarity 1.0 -> ``min_steps``, similarity 0.0 -> ``max_steps``.
    """
    if min_steps > max_steps:
        raise ValueError("min_steps must not exceed max_steps")
    s = _clip01(similarity)
    return int(round(min_steps + (1.0 - s) * (max_steps - min_steps)))


def fooling_probability(similarity: float, steepness: float = 10.0, midpoint: float = 0.6) -> float:
    """Logistic estimate of attack success, 0.5 at ``midpoint``.

    Increases with similarity: closer look-alikes are assumed easier to fool.
    """
    s = _clip01(similarity)
    return 1.0 / (1.0 + math.exp(-steepness * (s - midpoint)))


def similarity_tier(similarity: float) -> str:
    s = _clip01(similarity)
    if s >= HIGH_SIMILARITY:
        return "high"
    if s >= MEDIUM_SIMILARITY:
        return "medium"
    return "low"


def estimate_budget(
    similarity: float,
    min_steps: int = 5,
    max_steps: int = 40,
    steepness: float = 10.0,
    midpoint: float = 0.6,
) -> AttackBudget:
    return AttackBudget(
        similarity=_clip01(similarity),
        steps=similarity_to_steps(similarity, min_steps, max_steps),
        fooling_probability=fooling_probability(similarity, steepness, midpoint),
        tier=similarity_tier(similarity),
    )


def format_table(similarities: Iterable[float], **kwargs) -> str:
    rows = [f"{'similarity':>10}  {'tier':<6}  {'steps':>5}  {'p(fool)':>7}", "-" * 35]
    for sim in similarities:
        b = estimate_budget(sim, **kwargs)
        rows.append(f"{b.similarity:>10.3f}  {b.tier:<6}  {b.steps:>5d}  {b.fooling_probability:>7.1%}")
    return "\n".join(rows)
