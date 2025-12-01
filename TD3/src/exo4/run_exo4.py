import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import Counter, defaultdict
from helpers import entropy_from_counts, gini_from_counts

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

def dataset_entropy(rows):
    counts = Counter(r[2] for r in rows)
    return counts, entropy_from_counts(counts)


def info_gain_for_attribute(attr_index: int):
    """
    attr_index = 0 for A, 1 for B
    """
    parent_counts, parent_H = dataset_entropy(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[r[attr_index]].append(r)

    total_n = len(rows)
    child_H = 0.0
    details = {}

    for value, group in groups.items():
        counts_v, H_v = dataset_entropy(group)
        child_H += (len(group) / total_n) * H_v
        details[value] = (len(group), counts_v, H_v)

    gain = parent_H - child_H
    return parent_H, child_H, gain, details


def gini_dataset():
    counts, _ = dataset_entropy(rows)
    return gini_from_counts(counts), counts


def gini_gain_for_attribute(attr_index: int):
    """
    attr_index = 0 for A, 1 for B
    """
    parent_gini, parent_counts = gini_dataset()

    groups = defaultdict(list)
    for r in rows:
        groups[r[attr_index]].append(r)

    total_n = len(rows)
    child_gini = 0.0
    details = {}

    for value, group in groups.items():
        counts_v = Counter(r[2] for r in group)
        g_v = gini_from_counts(counts_v)
        child_gini += (len(group) / total_n) * g_v
        details[value] = (len(group), counts_v, g_v)

    gain = parent_gini - child_gini
    return parent_gini, child_gini, gain, details


def run_ex4():
    # (a) Entropy + info gain
    print("4(a) Entropy and information gain:")
    for idx, name in enumerate(["A", "B"]):
        parent_H, child_H, gain, details = info_gain_for_attribute(idx)
        print(f"  Attribute {name}:")
        print(f"    H(D) = {parent_H:.3f}")
        for v, (n, counts_v, H_v) in details.items():
            print(f"    {name} = {v}: n={n}, counts={counts_v}, H = {H_v:.3f}")
        print(f"    Weighted child entropy = {child_H:.3f}")
        print(f"    Gain({name}) = {gain:.3f}")
        print()

    # (b) Gini + Gini gain
    print("4(b) Gini and Gini gain:")
    parent_gini, parent_counts = gini_dataset()
    print(f"  Gini(D) = {parent_gini:.3f}, counts={parent_counts}")
    for idx, name in enumerate(["A", "B"]):
        parent_g, child_g, gain_g, details = gini_gain_for_attribute(idx)
        print(f"  Attribute {name}:")
        for v, (n, counts_v, g_v) in details.items():
            print(f"    {name} = {v}: n={n}, counts={counts_v}, Gini = {g_v:.3f}")
        print(f"    Weighted child Gini = {child_g:.3f}")
        print(f"    Gini gain({name}) = {gain_g:.3f}")
        print()

    print("4(c) Comparison:")
    print("  Info gain chooses A (higher gain).")
    print("  Gini gain chooses B (higher gain).")
    print("  So yes: the two criteria can favor different attributes.")


if __name__ == "__main__":
    run_ex4()
