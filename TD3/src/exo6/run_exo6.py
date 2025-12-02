from itertools import product
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

def generate_parity_dataset():
    header = ["A", "B", "C", "D", "Class"]
    rows = []

    for a, b, c, d in product([0, 1], repeat=4):
        s = a + b + c + d
        cls = 1 if s % 2 == 1 else 0
        rows.append([a, b, c, d, cls])

    return header, rows

def encode_for_sklearn(header, rows):
    X = np.array([r[:-1] for r in rows], dtype=float)
    y = np.array([r[-1] for r in rows], dtype=int)
    return X, y

def train_tree(X, y, criterion="entropy", random_state=0, max_depth=None):
    clf = DecisionTreeClassifier(
        criterion=criterion,
        random_state=random_state,
        max_depth=max_depth,
    )
    clf.fit(X, y)
    return clf


def save_tree_figure(clf, header, filename):
    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))
    tree.plot_tree(
        clf,
        feature_names=header[:-1],
        class_names=["0", "1"],
        impurity=True,
        filled=False,
        rounded=False,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {out_path}")

def main():
    # (a)
    header, rows = generate_parity_dataset()
    print("6(a) First 5 rows of parity dataset:")
    for r in rows[:5]:
        print("  ", r)
    print("  Total rows:", len(rows))
    print()

    # (b) encode + show type limitation
    X, y = encode_for_sklearn(header, rows)
    print("6(b) Data types for sklearn:")
    print("  X shape:", X.shape, "dtype:", X.dtype)
    print("  y shape:", y.shape, "dtype:", y.dtype)
    print("  -> fit() expects numeric features (no 'T'/'F' strings,")
    print("     we must encode Booleans / categories as numbers).")
    print()

    # (c) generate trees and save a figure
    print("6(c) Trees with different parameters:")
    clf_entropy = train_tree(X, y, criterion="entropy", random_state=0)
    print("  criterion='entropy', random_state=0, accuracy:",
          clf_entropy.score(X, y))
    save_tree_figure(clf_entropy, header,
                     "src/images/exo6/parity_tree_entropy.png")

    clf_gini = train_tree(X, y, criterion="gini", random_state=0)
    print("  criterion='gini',   random_state=0, accuracy:",
          clf_gini.score(X, y))

    clf_entropy_rs = train_tree(X, y, criterion="entropy", random_state=42)
    print("  criterion='entropy', random_state=42, accuracy:",
          clf_entropy_rs.score(X, y))

    clf_shallow = train_tree(X, y, criterion="entropy",
                             random_state=0, max_depth=2)
    print("  criterion='entropy', max_depth=2 (underfitting), accuracy:",
          clf_shallow.score(X, y))


if __name__ == "__main__":
    main()
