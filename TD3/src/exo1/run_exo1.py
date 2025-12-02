from pathlib import Path
from impurity_measures import entropy, gini, classificationError
import matplotlib.pyplot as plt

def test_impurities():
    test_probs = [
        [0.0, 1.0],
        [0.1667, 0.8333],
        [0.5, 0.5],
    ]

    for probs in test_probs:
        h = entropy(probs)
        g = gini(probs)
        e = classificationError(probs)
        print(f"probs = {probs}")
        print(f"  Entropy            = {h:.4f}")
        print(f"  Gini               = {g:.4f}")
        print(f"  Misclass. error    = {e:.4f}")
        print()


def plot_impurity_curves():
    out_path = Path("src/images/exo1/impurity_curves.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ps = [i / 100.0 for i in range(0, 101)]

    entropies = []
    ginis = []
    errors = []

    for p in ps:
        probs = [p, 1.0 - p]
        entropies.append(entropy(probs))
        ginis.append(gini(probs))
        errors.append(classificationError(probs))

    plt.figure()
    plt.plot(ps, entropies, label="Entropy")
    plt.plot(ps, ginis, linestyle="--", label="Gini")
    plt.plot(ps, errors, linestyle="-.", label="Misclassification error")
    plt.xlabel("p")
    plt.ylabel("Impurity")
    plt.legend()
    plt.title("Comparison of impurity measures (binary classification)")
    plt.grid(True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    test_impurities()
    plot_impurity_curves()