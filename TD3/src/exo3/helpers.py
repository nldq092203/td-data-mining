from collections import Counter
from exo1.impurity_measures import entropy, gini, classificationError

def entropy_from_counts(class_counts: Counter) -> float:
    n = sum(class_counts.values())
    probs = [c / n for c in class_counts.values()]
    return entropy(probs)


def gini_from_counts(class_counts: Counter) -> float:
    n = sum(class_counts.values())
    probs = [c / n for c in class_counts.values()]
    return gini(probs)


def classification_error_from_counts(class_counts: Counter) -> float:
    n = sum(class_counts.values())
    probs = [c / n for c in class_counts.values()]
    return classificationError(probs)