from collections import Counter, defaultdict
from exo1.impurity_measures import gini

def gini_of_counts(class_counts: Counter) -> float:
    n = sum(class_counts.values())
    probs = [c / n for c in class_counts.values()]
    return gini(probs)


def gini_dataset(dataset) -> float:
    header, rows = dataset
    class_idx = header.index("Class")
    counts = Counter(r[class_idx] for r in rows)
    return gini_of_counts(counts)


def gini_attribute_multiway(dataset, attr_name):
    """
    Multiway split on a categorical attribute.
    Returns: (gini_split, groups_dict)
    groups_dict maps each attribute value -> list of rows.
    """
    header, rows = dataset
    attr_idx = header.index(attr_name)
    class_idx = header.index("Class")

    groups = defaultdict(list)
    for r in rows:
        groups[r[attr_idx]].append(r)

    total_n = len(rows)
    split_gini = 0.0

    for value, rows_v in groups.items():
        counts = Counter(r[class_idx] for r in rows_v)
        g_child = gini_of_counts(counts)
        n_child = len(rows_v)
        split_gini += (n_child / total_n) * g_child

    return split_gini, groups


def gini_attribute_binary_splits(dataset, attr_name):
    """
    Try all binary splits of a categorical attribute and
    return list of (subset_values_tuple, gini_split).
    """
    from itertools import combinations

    header, rows = dataset
    attr_idx = header.index(attr_name)
    class_idx = header.index("Class")

    values = sorted({r[attr_idx] for r in rows})
    k = len(values)
    results = []

    def gini_of_rows(rs):
        counts = Counter(r[class_idx] for r in rs)
        return gini_of_counts(counts)

    for r in range(1, (k // 2) + 1):
        for subset in combinations(values, r):
            subset_set = set(subset)
            left = [row for row in rows if row[attr_idx] in subset_set]
            right = [row for row in rows if row[attr_idx] not in subset_set]

            g_left = gini_of_rows(left)
            g_right = gini_of_rows(right)

            n = len(rows)
            g_split = (len(left) / n) * g_left + (len(right) / n) * g_right
            results.append((tuple(sorted(subset)), g_split))

    return results