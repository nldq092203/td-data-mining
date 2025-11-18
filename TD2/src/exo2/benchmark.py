# src/exo2/benchmark.py

import time
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.exo2.generator import generate_transactions
from src.exo1.apriori import apriori_frequent_itemsets


def benchmark_single_param(
    n_items,
    width_range,
    minsup,
    transaction_sizes,
    n_runs=3,
    generator_kwargs=None,
):
    """
    Benchmark pour comparer les 3 méthodes Apriori
    en faisant varier uniquement le nombre de transactions.
    
    transaction_sizes : liste des tailles de datasets [1000, 2000, 3000]
    n_runs            : nombre de répétitions pour lisser les mesures
    generator_kwargs  : paramètres additionnels passés au générateur
    """

    methods = {
        "Brute-Force": "bruteforce",
        "F_{k-1} x F_1": "fk1_f1",
        "F_{k-1} x F_{k-1}": "fk1_fk1"
    }

    results = {m: [] for m in methods}

    for n_trans in transaction_sizes:
        print(f"\n--- Benchmark n={n_trans} ---")
        trans = generate_transactions(
            n_transactions=n_trans,
            n_items=n_items,
            width_range=width_range,
            **(generator_kwargs or {}),
        )

        for label, method in methods.items():
            times = []

            for _ in range(n_runs):
                t0 = time.time()
                # Skip pruning for the brute-force baseline so that it
                # counts every possible candidate and exposes the cost
                # difference visible in the benchmarks.
                apriori_frequent_itemsets(
                    trans,
                    minsup=minsup,
                    method=method,
                    verbose=False,
                    prune=(method != "bruteforce")
                )
                t1 = time.time()
                times.append(t1 - t0)

            avg = sum(times) / len(times)
            results[label].append(avg)
            print(f"{label}: {avg:.4f} s")

    return results


def plot_results(transaction_sizes, results, title):
    """
    Génère une figure matplotlib pour les temps obtenus.
    """

    plt.figure(figsize=(10, 4))

    for label, values in results.items():
        plt.plot(
            transaction_sizes,
            values,
            marker="o",
            label=label
        )

    plt.xlabel("Nombre de transactions")
    plt.ylabel("Temps (secondes)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Paramètres du générateur
    n_items = 200
    width_range = (25, 30)
    minsup = 0.25
    sizes = [1000, 2000, 3000]
    generator_kwargs = {
        "popular_ratio": 0.15,   # 15% des items sont populaires
        "popular_weight": 12.0,  # ils apparaissent beaucoup plus souvent
    }

    results = benchmark_single_param(
        n_items=n_items,
        width_range=width_range,
        minsup=minsup,
        transaction_sizes=sizes,
        n_runs=3,
        generator_kwargs=generator_kwargs,
    )

    plot_results(
        sizes,
        results,
        title="Comparaison des méthodes Apriori"
    )

    results2 = benchmark_single_param(
        n_items=n_items,
        width_range=width_range,
        minsup=minsup,
        transaction_sizes=sizes,
        n_runs=3,
        generator_kwargs=None,  # Pas d'items populaires
    )   

    plot_results(
        sizes,
        results2,
        title="Comparaison des méthodes Apriori (dataset équilibré)"
    )
