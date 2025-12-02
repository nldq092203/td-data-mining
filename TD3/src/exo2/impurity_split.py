from exo1.impurity_measures import gini

def impurity_split(cnt, impurity_function):
    """
    cnt: dict like {"N1": {"C0":4, "C1":3}, "N2": {"C0":2, "C1":3}}
    impurity_function: e.g. gini, entropy, classificationError
    returns weighted impurity of the split
    """
    total = sum(sum(class_counts.values()) for class_counts in cnt.values())
    result = 0.0

    for node_name, class_counts in cnt.items():
        n_i = sum(class_counts.values())
        probs = [c / n_i for c in class_counts.values()]
        result += (n_i / total) * impurity_function(probs)

    return result


def test_impurity_split():
    d1 = {"N1": {"C0": 4, "C1": 3}, "N2": {"C0": 2, "C1": 3}}
    d2 = {"N1": {"C0": 1, "C1": 4}, "N2": {"C0": 5, "C1": 2}}

    print("2(a) Tests for impurity_split with Gini:")
    print("d1 =", d1)
    print("  impurity_split(d1, gini) =", round(impurity_split(d1, gini), 3))
    print("d2 =", d2)
    print("  impurity_split(d2, gini) =", round(impurity_split(d2, gini), 3))
    print()