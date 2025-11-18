import time
from apriori import frequent_itemsets_gen
import matplotlib.pyplot as plt
import random


def generate_synthetic_transactions(n_transactions, n_items, min_width, max_width):
    random.seed(0)
    items = [f"p{i}" for i in range(n_items)]
    transactions = []

    for _ in range(n_transactions):
        width = random.randint(min_width, max_width)
        t = random.sample(items, width)
        transactions.append(t)

    return transactions


def benchmark(transactions, minsup, n_runs=5):
    methods = ["bruteforce", "Fk-1xF1", "Fk-1xFk-1"]
    results = {}

    for m in methods:
        times = []
        for _ in range(n_runs):
            start = time.time()
            frequent_itemsets_gen(transactions, minsup, method=m, verbose=False)
            times.append(time.time() - start)
        # moyenne
        results[m] = sum(times) / len(times)

    return results


def run_experiment_vary_transactions(
    list_n_transactions, n_items, min_width, max_width, minsup
):
    brute, fk1, fkfk = [], [], []

    for n in list_n_transactions:
        print(f"Running n_transactions = {n}")
        transactions = generate_synthetic_transactions(n, n_items, min_width, max_width)
        times = benchmark(transactions, minsup)

        brute.append(times["bruteforce"])
        fk1.append(times["Fk-1xF1"])
        fkfk.append(times["Fk-1xFk-1"])

    return brute, fk1, fkfk


def plot_results(x_values, brute, fk1, fkfk, title):
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, brute, label="Brute-force", marker="o")
    plt.plot(x_values, fk1, label="F(k-1) × F1", marker="o")
    plt.plot(x_values, fkfk, label="F(k-1) × F(k-1)", marker="o")
    plt.xlabel("Number of Transactions")
    plt.ylabel("Execution Time (s)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


list_n = [10, 100, 1000]
brute, fk1, fkfk = run_experiment_vary_transactions(
    list_n_transactions=list_n, n_items=50, min_width=5, max_width=15, minsup=0.5
)

plot_results(
    list_n, brute, fk1, fkfk, "Small dataset: 3 methods are close to each other"
)
