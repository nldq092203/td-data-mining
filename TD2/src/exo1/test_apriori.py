from apriori import apriori_frequent_itemsets

def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")

data_a = [
    ["a", "b", "d", "e"],
    ["b", "c", "d"],
    ["a", "b", "d", "e"],
    ["a", "c", "d", "e"],
    ["b", "c", "d", "e"],
    ["b", "d", "e"],
    ["c", "d"],
    ["a", "b", "c"],
    ["a", "d", "e"],
    ["b", "d"],
]

minsup_a = 0.3

data_b = [
    ["b", "c", "d"],
    ["a", "b", "c", "d", "e"],
    ["a", "b", "c", "e"],
    ["a", "b", "d", "e"],
    ["b", "c", "e"],
    ["a", "b", "d", "e"],
]

minsup_b = 0.5

def test_dataset(transactions, minsup, dataset_name):
    header(f"Testing dataset {dataset_name} — minsup={minsup}")

    methods = [
        ("Méthode Brute-force", "bruteforce"),
        ("Méthode F(k-1) x F1", "fk1_f1"),
        ("Méthode F(k-1) x F(k-1)", "fk1_fk1"),
    ]

    for label, method in methods:
        print("\n" + label)
        apriori_frequent_itemsets(
            transactions, minsup, method=method, verbose=True
        )

if __name__ == "__main__":
    test_dataset(data_a, minsup_a, "(a)")
    test_dataset(data_b, minsup_b, "(b)")
