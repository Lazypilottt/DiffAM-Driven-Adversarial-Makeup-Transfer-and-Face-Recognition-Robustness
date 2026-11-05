"""Similarity-guided adversarial makeup attacks built on DiffAM."""

from .budget import AttackBudget, estimate_budget, fooling_probability, similarity_to_steps

__all__ = ["AttackBudget", "estimate_budget", "fooling_probability", "similarity_to_steps"]
__version__ = "0.1.0"
