from math import log
import matplotlib.pyplot as plt

def entropy(probs):
    """
    probs: list of probabilities that sum to 1.
    Uses log base 2.
    """
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * log(p, 2)
    return h


def gini(probs):
    """
    Gini impurity: 1 - sum_j p_j^2
    """
    return 1.0 - sum(p * p for p in probs)


def classificationError(probs):
    """
    Misclassification error: 1 - max_j p_j
    """
    return 1.0 - max(probs)