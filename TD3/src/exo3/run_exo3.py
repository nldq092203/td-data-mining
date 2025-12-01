import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

header = ["Instance", "a1", "a2", "a3", "Target Class"]
rows = [
    ["1", "T", "T", 1.0, "+"],
    ["2", "T", "T", 6.0, "+"],
    ["3", "T", "F", 5.0, "-"],
    ["4", "F", "F", 4.0, "+"],
    ["5", "F", "T", 7.0, "-"],
    ["6", "F", "T", 3.0, "-"],
    ["7", "F", "F", 8.0, "-"],
    ["8", "T", "F", 7.0, "+"],
    ["9", "F", "T", 5.0, "-"],
]

dataset = (header, rows)

from collections import Counter, defaultdict
from helpers import entropy_from_counts, gini_from_counts, classification_error_from_counts

def dataset_entropy(dataset):
    header, rows = dataset
    class_idx = header.index("Target Class")
    counts = Counter(r[class_idx] for r in rows)
    return entropy_from_counts(counts), counts

def info_gain_categorical(dataset, attr_name):
    header, rows = dataset
    attr_idx = header.index(attr_name)
    class_idx = header.index("Target Class")

    # parent entropy
    parent_entropy, parent_counts = dataset_entropy(dataset)

    # split by attribute value
    groups = defaultdict(list)
    for r in rows:
        groups[r[attr_idx]].append(r)

    total_n = len(rows)
    child_entropy = 0.0
    details = {}

    for value, rows_v in groups.items():
        counts_v = Counter(r[class_idx] for r in rows_v)
        h_v = entropy_from_counts(counts_v)
        child_entropy += (len(rows_v) / total_n) * h_v
        details[value] = (counts_v, h_v)

    gain = parent_entropy - child_entropy
    return parent_entropy, child_entropy, gain, details

def info_gain_continuous(dataset, attr_name):
    header, rows = dataset
    attr_idx = header.index(attr_name)
    class_idx = header.index("Target Class")

    parent_entropy, parent_counts = dataset_entropy(dataset)

    # sort rows by continuous value
    rows_sorted = sorted(rows, key=lambda r: r[attr_idx])
    values = sorted({r[attr_idx] for r in rows_sorted})
    thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]

    results = []

    for t in thresholds:
        left = [r for r in rows_sorted if r[attr_idx] <= t]
        right = [r for r in rows_sorted if r[attr_idx] > t]

        counts_left = Counter(r[class_idx] for r in left)
        counts_right = Counter(r[class_idx] for r in right)

        h_left = entropy_from_counts(counts_left)
        h_right = entropy_from_counts(counts_right)

        n_total = len(rows_sorted)
        split_entropy = (len(left) / n_total) * h_left + (len(right) / n_total) * h_right
        gain = parent_entropy - split_entropy

        results.append(
            {
                "threshold": t,
                "split_entropy": split_entropy,
                "gain": gain,
                "left_counts": counts_left,
                "right_counts": counts_right,
            }
        )

    return parent_entropy, results

def impurity_split_categorical(dataset, attr_name, impurity_func):
    header, rows = dataset
    attr_idx = header.index(attr_name)
    class_idx = header.index("Target Class")

    groups = defaultdict(list)
    for r in rows:
        groups[r[attr_idx]].append(r)

    total_n = len(rows)
    split_imp = 0.0
    details = {}

    for value, rows_v in groups.items():
        counts_v = Counter(r[class_idx] for r in rows_v)
        imp_v = impurity_func(counts_v)
        split_imp += (len(rows_v) / total_n) * imp_v
        details[value] = (counts_v, imp_v)

    return split_imp, details

def run_ex3():
    # (a)
    H_D, counts = dataset_entropy(dataset)
    print("3(a) Entropy of dataset:")
    print("  class counts:", counts)
    print(f"  H(D) = {H_D:.3f}")
    print()

    # (b)
    for attr in ["a1", "a2"]:
        print(f"3(b) Info gain for {attr}:")
        parent_H, child_H, gain, details = info_gain_categorical(dataset, attr)
        print(f"  parent entropy H(D) = {parent_H:.3f}")
        for value, (c, h) in details.items():
            print(f"  {attr} = {value}: counts={c}, H = {h:.3f}")
        print(f"  sum weighted child entropies = {child_H:.3f}")
        print(f"  Gain({attr}) = {gain:.3f}")
        print()

    # (c)
    print("3(c) Info gain for continuous attribute a3 (all possible thresholds):")
    parent_H, results = info_gain_continuous(dataset, "a3")
    print(f"  parent entropy H(D) = {parent_H:.3f}")
    for r in results:
        t = r["threshold"]
        se = r["split_entropy"]
        g = r["gain"]
        lc = r["left_counts"]
        rc = r["right_counts"]
        print(
            f"  threshold a3 <= {t:.1f}: "
            f"left={lc}, right={rc}, H_split={se:.3f}, Gain={g:.3f}"
        )
    print()

    # (d) best split by info gain
    _, _, gain_a1, _ = info_gain_categorical(dataset, "a1")
    _, _, gain_a2, _ = info_gain_categorical(dataset, "a2")
    _, results_a3 = info_gain_continuous(dataset, "a3")
    best_a3 = max(results_a3, key=lambda r: r["gain"])
    print("3(d) Best split according to information gain:")
    print(f"  Gain(a1) = {gain_a1:.3f}")
    print(f"  Gain(a2) = {gain_a2:.3f}")
    print(
        f"  Best Gain(a3) = {best_a3['gain']:.3f} "
        f"for threshold a3 <= {best_a3['threshold']:.1f}"
    )
    print("  -> Best attribute: a1 (highest information gain).")
    print()

    # (e) best split (a1 vs a2) by misclassification error
    err_a1, _ = impurity_split_categorical(
        dataset, "a1", classification_error_from_counts
    )
    err_a2, _ = impurity_split_categorical(
        dataset, "a2", classification_error_from_counts
    )
    print("3(e) Best split (a1 vs a2) by classification error:")
    print(f"  Error split(a1) = {err_a1:.3f}")
    print(f"  Error split(a2) = {err_a2:.3f}")
    print("  -> Best (lowest error): a1")
    print()

    # (f) best split (a1 vs a2) by Gini index
    gini_a1, _ = impurity_split_categorical(dataset, "a1", gini_from_counts)
    gini_a2, _ = impurity_split_categorical(dataset, "a2", gini_from_counts)
    print("3(f) Best split (a1 vs a2) by Gini index:")
    print(f"  Gini split(a1) = {gini_a1:.3f}")
    print(f"  Gini split(a2) = {gini_a2:.3f}")
    print("  -> Best (lowest Gini): a1")


if __name__ == "__main__":
    run_ex3()
