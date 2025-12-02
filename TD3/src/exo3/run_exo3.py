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

from collections import Counter
from exo1.impurity_measures import entropy, gini, classificationError
from exo2.helpers import dataset_impurity, attribute_multiway_split, impurity_of_counts

def info_gain_categorical(dataset, attr_name):
    """
    Calculate information gain for a categorical attribute using entropy.
    """
    parent_entropy = dataset_impurity(dataset, entropy, class_attr="Target Class")

    split_entropy, groups = attribute_multiway_split(dataset, attr_name, entropy, class_attr="Target Class")

    gain = parent_entropy - split_entropy

    header, rows = dataset
    class_idx = header.index("Target Class")
    details = {}
    for value, rows_v in groups.items():
        counts_v = Counter(r[class_idx] for r in rows_v)
        h_v = impurity_of_counts(counts_v, entropy)
        details[value] = (counts_v, h_v)

    return parent_entropy, split_entropy, gain, details

def info_gain_continuous(dataset, attr_name):
    """
    Calculate information gain for a continuous attribute by testing all thresholds.
    """
    header, rows = dataset
    attr_idx = header.index(attr_name)
    class_idx = header.index("Target Class")

    parent_entropy = dataset_impurity(dataset, entropy, class_attr="Target Class")

    rows_sorted = sorted(rows, key=lambda r: r[attr_idx])
    values = sorted({r[attr_idx] for r in rows_sorted})
    thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]

    results = []

    for t in thresholds:
        left = [r for r in rows_sorted if r[attr_idx] <= t]
        right = [r for r in rows_sorted if r[attr_idx] > t]

        counts_left = Counter(r[class_idx] for r in left)
        counts_right = Counter(r[class_idx] for r in right)

        h_left = impurity_of_counts(counts_left, entropy)
        h_right = impurity_of_counts(counts_right, entropy)

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

def run_ex3():
    # (a)
    H_D = dataset_impurity(dataset, entropy, class_attr="Target Class")
    print("3(a) Entropy of dataset:")
    print(f"  entropy(dataset) = {H_D:.3f}")
    print()

    # (b)
    for attr in ["a1", "a2"]:
        print(f"3(b) Info gain for {attr}:")
        parent_entropy, child_entropy, gain, details = info_gain_categorical(dataset, attr)
        print(f"  entropy(dataset) = {parent_entropy:.3f}")
        for value, (c, h) in details.items():
            print(f"  {attr} = {value}: counts={c}, entropy = {h:.3f}")
        print(f"  entropy after split on {attr} = {child_entropy:.3f}")
        print(f"  gain({attr}) = {gain:.3f}")
        print()

    # (c)
    print("3(c) Info gain for continuous attribute a3 (all possible thresholds):")
    parent_entropy, results = info_gain_continuous(dataset, "a3")
    print(f"  entropy(dataset) = {parent_entropy:.3f}")
    for r in results:
        t = r["threshold"]
        se = r["split_entropy"]
        g = r["gain"]
        lc = r["left_counts"]
        rc = r["right_counts"]
        print(
            f"  threshold a3 <= {t:.1f}: "
            f"left={lc}, right={rc}, entropy_split={se:.3f}, gain={g:.3f}"
        )
    print()

    # (d) best split by info gain
    _, _, gain_a1, _ = info_gain_categorical(dataset, "a1")
    _, _, gain_a2, _ = info_gain_categorical(dataset, "a2")
    _, results_a3 = info_gain_continuous(dataset, "a3")
    best_a3 = max(results_a3, key=lambda r: r["gain"])
    print("3(d) Best split according to information gain:")
    print(f"  gain(a1) = {gain_a1:.3f}")
    print(f"  gain(a2) = {gain_a2:.3f}")
    print(
        f"  Best gain(a3) = {best_a3['gain']:.3f} "
        f"for threshold a3 <= {best_a3['threshold']:.1f}"
    )
    print("  -> Best attribute: a1 (highest information gain).")
    print()

    # (e) best split (a1 vs a2) by misclassification error
    err_a1, _ = attribute_multiway_split(
        dataset, "a1", classificationError, class_attr="Target Class"
    )
    err_a2, _ = attribute_multiway_split(
        dataset, "a2", classificationError, class_attr="Target Class"
    )
    print("3(e) Best split (a1 vs a2) by classification error:")
    print(f"  error split(a1) = {err_a1:.3f}")
    print(f"  error split(a2) = {err_a2:.3f}")
    print("  -> Best (lowest error): a1")
    print()

    # (f) best split (a1 vs a2) by Gini index
    gini_a1, _ = attribute_multiway_split(dataset, "a1", gini, class_attr="Target Class")
    gini_a2, _ = attribute_multiway_split(dataset, "a2", gini, class_attr="Target Class")
    print("3(f) Best split (a1 vs a2) by Gini index:")
    print(f"  gini split(a1) = {gini_a1:.3f}")
    print(f"  gini split(a2) = {gini_a2:.3f}")
    print("  -> Best (lowest Gini): a1")


if __name__ == "__main__":
    run_ex3()
