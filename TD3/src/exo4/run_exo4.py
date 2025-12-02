import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import Counter
from exo1.impurity_measures import entropy, gini
from exo2.helpers import dataset_impurity, attribute_multiway_split, impurity_of_counts

# ---------- Dataset ----------
header = ["A", "B", "Class Label"]
rows = [
    ["T", "F", "+"],
    ["T", "T", "+"],
    ["T", "T", "+"],
    ["T", "F", "-"],
    ["T", "T", "+"],
    ["F", "F", "-"],
    ["F", "F", "-"],
    ["F", "F", "-"],
    ["T", "T", "-"],
    ["T", "F", "-"],
]

dataset = (header, rows)


def info_gain_for_attribute(attr_name: str):
    """
    Calculate information gain for an attribute using entropy.
    """
    parent_H = dataset_impurity(dataset, entropy, class_attr="Class Label")
    child_H, groups = attribute_multiway_split(dataset, attr_name, entropy, class_attr="Class Label")

    gain = parent_H - child_H

    # Build details dict
    class_idx = header.index("Class Label")
    details = {}
    for value, rows_v in groups.items():
        counts_v = Counter(r[class_idx] for r in rows_v)
        H_v = impurity_of_counts(counts_v, entropy)
        details[value] = (len(rows_v), counts_v, H_v)

    return parent_H, child_H, gain, details


def gini_gain_for_attribute(attr_name: str):
    """
    Calculate Gini gain for an attribute.
    """
    parent_gini = dataset_impurity(dataset, gini, class_attr="Class Label")
    child_gini, groups = attribute_multiway_split(dataset, attr_name, gini, class_attr="Class Label")

    gain = parent_gini - child_gini

    # Build details dict
    class_idx = header.index("Class Label")
    details = {}
    for value, rows_v in groups.items():
        counts_v = Counter(r[class_idx] for r in rows_v)
        g_v = impurity_of_counts(counts_v, gini)
        details[value] = (len(rows_v), counts_v, g_v)

    return parent_gini, child_gini, gain, details


def run_ex4():
    # (a) Entropy + info gain
    print("4(a) Entropy and information gain:")
    for name in ["A", "B"]:
        parent_H, child_H, gain, details = info_gain_for_attribute(name)
        print(f"  Attribute {name}:")
        print(f"    entropy(dataset) = {parent_H:.3f}")
        for v, (n, counts_v, H_v) in details.items():
            print(f"    {name} = {v}: n={n}, counts={counts_v}, entropy = {H_v:.3f}")
        print(f"    sum weighted child entropies = {child_H:.3f}")
        print(f"    gain({name}) = {gain:.3f}")
        print()

    # (b) Gini + Gini gain
    print("4(b) Gini and Gini gain:")
    parent_gini = dataset_impurity(dataset, gini, class_attr="Class Label")
    class_idx = header.index("Class Label")
    parent_counts = Counter(r[class_idx] for r in rows)
    print(f"  gini(dataset) = {parent_gini:.3f}, counts={parent_counts}")
    for name in ["A", "B"]:
        parent_g, child_g, gain_g, details = gini_gain_for_attribute(name)
        print(f"  Attribute {name}:")
        for v, (n, counts_v, g_v) in details.items():
            print(f"    {name} = {v}: n={n}, counts={counts_v}, gini = {g_v:.3f}")
        print(f"    sum weighted child gini = {child_g:.3f}")
        print(f"    gini gain({name}) = {gain_g:.3f}")
        print()

    print("4(c) Comparison:")
    print("  Info gain chooses A (higher gain).")
    print("  Gini gain chooses B (higher gain).")
    print("  So yes: the two criteria can favor different attributes.")


if __name__ == "__main__":
    run_ex4()
