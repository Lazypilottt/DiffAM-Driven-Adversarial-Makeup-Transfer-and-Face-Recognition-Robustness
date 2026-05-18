def similarity_to_noise(similarity, min_noise=5, max_noise=40):
def similarity_to_fooling_rate(similarity, steepness=10, mid_point=0.6):
def estimate_noise_and_fooling(similarity_scores):

"""
Similarity-to-Noise and Fooling Rate Estimator
----------------------------------------------
This script provides utility functions to estimate the required noise level and the probability of fooling a face recognition system based on embedding similarity.
"""

import numpy as np

def similarity_to_noise(similarity, min_noise=5, max_noise=40):
    """
    Map similarity [0,1] to a noise budget.
    Higher similarity means less noise is needed to fool the system.
    """
    similarity = np.clip(similarity, 0, 1)
    noise = (1 - similarity) * (max_noise - min_noise) + min_noise
    return round(noise, 2)

def similarity_to_fooling_rate(similarity, steepness=10, mid_point=0.6):
    """
    Estimate the fooling rate using a logistic curve.
    Higher similarity means a higher chance of fooling with less noise.
    """
    x = 1 - similarity
    rate = 1 / (1 + np.exp(-steepness * (x - (1 - mid_point))))
    return round(rate * 100, 2)  # as a percentage

def estimate_noise_and_fooling(similarity_scores):
    """
    Print a table of similarity, estimated noise, and fooling rate.
    """
    print(f"{'Similarity':<12} {'Noise_Level':<12} {'Fooling_Rate(%)':<16}")
    print("-" * 40)
    for sim in similarity_scores:
        noise = similarity_to_noise(sim)
        fooling_rate = similarity_to_fooling_rate(sim)
        print(f"{sim:<12.4f} {noise:<12} {fooling_rate:<16}")

if __name__ == "__main__":
    # Example similarities (replace with real embedding comparisons as needed)
    sample_similarities = [0.95, 0.85, 0.75, 0.60, 0.45, 0.30]
    estimate_noise_and_fooling(sample_similarities)
