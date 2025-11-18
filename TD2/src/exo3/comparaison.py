import random
import time

import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
import numpy as np
import pandas as pd
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession

from exo1.apriori import apriori_frequent_itemsets


def generate_synthetic_transactions(n_transactions, n_items, min_width, max_width):
    """Generate synthetic transactions with random item subsets."""
    items = [f"p{i}" for i in range(n_items)]
    max_width = min(max_width, n_items)

    transactions = []
    for _ in range(n_transactions):
        width = random.randint(min_width, max_width)
        transactions.append(random.sample(items, width))

    return transactions


# =========================================================
# My Apriori Implementation
# =========================================================
def run_my_apriori(transactions):
    start = time.time()
    apriori_frequent_itemsets(transactions, minsup=0.3, method="fk1_fk1", verbose=False)
    return time.time() - start


# =========================================================
# MLxtend Apriori
# =========================================================
def run_mlxtend(transactions):
    all_items = sorted({item for t in transactions for item in t})
    df = pd.DataFrame([{item: (item in t) for item in all_items} for t in transactions])

    start = time.time()
    apriori(df, min_support=0.3, use_colnames=True)
    return time.time() - start


# =========================================================
# Spark Session + Warm-up
# =========================================================
def prepare_df_spark(transactions):
    return spark.createDataFrame([(t,) for t in transactions], ["items"])


spark = SparkSession.builder.appName("Benchmark FP-Growth").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


def spark_full_warmup():
    print("Spark full warm-up ...")
    tx = generate_synthetic_transactions(
        n_transactions=5000,
        n_items=800,
        min_width=20,
        max_width=40,
    )
    df = prepare_df_spark(tx)
    fp = FPGrowth(itemsCol="items", minSupport=0.2, minConfidence=0.0)
    fp.fit(df)
    print("Spark warm-up done!")


spark_full_warmup()


def run_spark(df):
    fp = FPGrowth(itemsCol="items", minSupport=0.3, minConfidence=0.0)
    start = time.time()
    fp.fit(df)
    return time.time() - start


# =========================================================
# Benchmark Function
# =========================================================
def run_benchmark(title, n_items, min_width, max_width, filename):
    sizes = [1000, 2000, 3000]
    times_my = []
    times_ml = []
    times_sp = []
    n_runs = 3

    print(f"\n========== {title} ==========\n")

    for n in sizes:
        print(f"\n=== Running for N = {n} transactions ===")

        tx = generate_synthetic_transactions(n, n_items, min_width, max_width)
        df_spark = prepare_df_spark(tx)

        t_my = np.mean([run_my_apriori(tx) for _ in range(n_runs)])
        t_ml = np.mean([run_mlxtend(tx) for _ in range(n_runs)])
        t_sp = np.mean([run_spark(df_spark) for _ in range(n_runs)])

        times_my.append(t_my)
        times_ml.append(t_ml)
        times_sp.append(t_sp)

        print(f"My Apriori : {t_my:.4f} s")
        print(f"MLxtend   : {t_ml:.4f} s")
        print(f"Spark     : {t_sp:.4f} s")

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(sizes, times_my, marker="o", label="Mon implem")
    plt.plot(sizes, times_ml, marker="o", label="MLxtend")
    plt.plot(sizes, times_sp, marker="o", label="Spark")
    plt.xlabel("Nombre de transactions")
    plt.ylabel("Temps (secondes)")
    plt.legend()
    plt.title(title)

    # SAVE TO FILE
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved chart to: {filename}")


def scenario1():
    run_benchmark(
        title="Scenario 1 — My Apriori faster than Spark",
        n_items=2000,
        min_width=10,
        max_width=300,
        filename="../static/scenario1.png",
    )


def scenario1_2():
    run_benchmark(
        title="Scenario 1.2 — Apriori grows faster than Spark",
        n_items=2000,
        min_width=1,
        max_width=10,
        filename="../static/scenario2.png",
    )


def scenario2():
    run_benchmark(
        title="Scenario 2 — Spark faster than My Apriori",
        n_items=7000,
        min_width=50,
        max_width=60,
        filename="../static/scenario3.png",
    )


scenario1()
scenario1_2()
scenario2()
