import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from math import log, ceil

left_tree = ["C1", ["C2", "C3"]]
right_tree = [["C1", ["C1", "C2"]], ["C2", "C3"]]

def count_nodes(tree):
    """
    Returns (nb_internal_nodes, nb_leaves)
    tree: either a string (leaf) or [left_subtree, right_subtree] for an internal node.
    """
    if isinstance(tree, str):
        # leaf
        return 0, 1
    else:
        # internal node
        left, right = tree
        int_left, leaf_left = count_nodes(left)
        int_right, leaf_right = count_nodes(right)
        nb_internal = 1 + int_left + int_right
        nb_leaves = leaf_left + leaf_right
        return nb_internal, nb_leaves


def Cost_tree(tree, m, k):
    """
    Cost(tree) in bits.
    m: number of attributes
    k: number of classes
    """
    nb_internal, nb_leaves = count_nodes(tree)
    bits_attr = ceil(log(m, 2))   # cost to encode one attribute id
    bits_class = ceil(log(k, 2))  # cost to encode one class id
    return nb_internal * bits_attr + nb_leaves * bits_class


def Cost_data_given_tree(nb_errors, n):
    """
    Cost(data | tree) in bits.
    nb_errors: number of misclassified training examples by the tree
    n: total number of training tuples
    """
    bits_per_error = ceil(log(n, 2))
    return nb_errors * bits_per_error


def Total_cost(tree, m, k, nb_errors, n):
    return Cost_tree(tree, m, k) + Cost_data_given_tree(nb_errors, n)


def test_example():
    m = 16  # attributes
    k = 3   # classes

    # numbers of errors from the statement
    errors_left = 7
    errors_right = 4

    for n in [8, 16, 32, 64]:
        c_left = Total_cost(left_tree, m, k, errors_left, n)
        c_right = Total_cost(right_tree, m, k, errors_right, n)
        print(f"n = {n}")
        print(f"  Total cost left tree  = {c_left} bits")
        print(f"  Total cost right tree = {c_right} bits")
        print()


if __name__ == "__main__":
    # show Cost(tree) alone
    m, k = 16, 3
    print("Internal / leaves:")
    print("  left tree :", count_nodes(left_tree))
    print("  right tree:", count_nodes(right_tree))
    print()

    print("Cost(tree) only:")
    print("  left tree  =", Cost_tree(left_tree, m, k))
    print("  right tree =", Cost_tree(right_tree, m, k))
    print()

    # Example total costs for different n
    test_example()
