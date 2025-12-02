from collections import Counter, defaultdict

def impurity_of_counts(class_counts: Counter, impurity_function) -> float:
    """
    Calculate impurity from class counts using any impurity function.

    Args:
        class_counts: Counter object with class labels as keys and counts as values
        impurity_function: Function that takes a list of probabilities and returns impurity

    Returns:
        Impurity value
    """
    n = sum(class_counts.values())
    if n == 0:
        return 0.0
    probs = [c / n for c in class_counts.values()]
    return impurity_function(probs)


def dataset_impurity(dataset, impurity_function, class_attr="Class"):
    """
    Calculate impurity of entire dataset using any impurity function.

    Args:
        dataset: Tuple of (header, rows)
        impurity_function: Function that takes a list of probabilities and returns impurity
        class_attr: Name of the class attribute column (default: "Class")

    Returns:
        Impurity value for the dataset
    """
    header, rows = dataset
    class_idx = header.index(class_attr)
    counts = Counter(r[class_idx] for r in rows)
    return impurity_of_counts(counts, impurity_function)


def attribute_multiway_split(dataset, attr_name, impurity_function, class_attr="Class"):
    """
    Calculate impurity after multiway split on a categorical attribute.
    Works with any impurity function.

    Args:
        dataset: Tuple of (header, rows)
        attr_name: Name of attribute to split on
        impurity_function: Function that takes a list of probabilities and returns impurity
        class_attr: Name of the class attribute column (default: "Class")

    Returns:
        (split_impurity, groups_dict) where groups_dict maps each attribute value -> list of rows
    """
    header, rows = dataset
    attr_idx = header.index(attr_name)
    class_idx = header.index(class_attr)

    groups = defaultdict(list)
    for r in rows:
        groups[r[attr_idx]].append(r)

    total_n = len(rows)
    split_impurity = 0.0

    for value, rows_v in groups.items():
        counts = Counter(r[class_idx] for r in rows_v)
        impurity_child = impurity_of_counts(counts, impurity_function)
        n_child = len(rows_v)
        split_impurity += (n_child / total_n) * impurity_child

    return split_impurity, groups


def attribute_binary_splits(dataset, attr_name, impurity_function, class_attr="Class"):
    """
    Try all binary splits of a categorical attribute and return list of results.
    Works with any impurity function.

    Args:
        dataset: Tuple of (header, rows)
        attr_name: Name of attribute to split on
        impurity_function: Function that takes a list of probabilities and returns impurity
        class_attr: Name of the class attribute column (default: "Class")

    Returns:
        List of (subset_values_tuple, split_impurity) tuples
    """
    from itertools import combinations

    header, rows = dataset
    attr_idx = header.index(attr_name)
    class_idx = header.index(class_attr)

    values = sorted({r[attr_idx] for r in rows})
    k = len(values)
    results = []

    def impurity_of_rows(rs):
        counts = Counter(r[class_idx] for r in rs)
        return impurity_of_counts(counts, impurity_function)

    for r in range(1, (k // 2) + 1):
        for subset in combinations(values, r):
            subset_set = set(subset)
            left = [row for row in rows if row[attr_idx] in subset_set]
            right = [row for row in rows if row[attr_idx] not in subset_set]

            impurity_left = impurity_of_rows(left)
            impurity_right = impurity_of_rows(right)

            n = len(rows)
            split_impurity = (len(left) / n) * impurity_left + (len(right) / n) * impurity_right
            results.append((tuple(sorted(subset)), split_impurity))

    return results