from collections import Counter
from exo1.impurity_measures import entropy, gini, classificationError

def entropy_from_counts(counts: Counter) -> float:
    n = sum(counts.values())
    probs = [c / n for c in counts.values()]
    return entropy(probs)


def gini_from_counts(counts: Counter) -> float:
    n = sum(counts.values())
    probs = [c / n for c in counts.values()]
    return gini(probs)