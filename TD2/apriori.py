import itertools


def normalize_transactions(transactions):
    return [set(t) for t in transactions]


def compute_support(candidates, transactions):
    counts = {c: 0 for c in candidates}
    N = len(transactions)

    for t in transactions:
        for c in candidates:
            if c.issubset(t):
                counts[c] += 1

    supports = {c: counts[c] / N for c in candidates}
    return supports


def generate_C1(transactions):
    items = set()
    for t in transactions:
        items.update(t)

    C1 = {frozenset([i]) for i in items}
    return C1


def verbose_print_C1(C1, support_C1, minsup, N):
    print("1-itemsets")
    for itemset in sorted(C1, key=lambda x: list(x)):
        item = list(itemset)[0]
        count = int(support_C1[itemset] * N)
        label = "F" if support_C1[itemset] >= minsup else "I"
        print(f"{item} {count} {label}")
    print("=========================")


def verbose_print_Ck(k, Ck, support_Ck, minsup, N, pruned):
    print(f"{k}-itemsets")

    # print N candidates first
    for itemset in sorted(pruned, key=lambda x: sorted(x)):
        items = ",".join(sorted(itemset))
        print(f"{items} N")

    # print counted candidates (I/F)
    for itemset in sorted(Ck - pruned, key=lambda x: sorted(x)):
        items = ",".join(sorted(itemset))
        count = int(support_Ck[itemset] * N)
        label = "F" if support_Ck[itemset] >= minsup else "I"
        print(f"{items} {count} {label}")

    print("=========================")


# BRUTE FORCE METHODS
def generate_Ck_bruteforce(all_items, k):
    return {frozenset(c) for c in itertools.combinations(all_items, k)}


#  F(k-1) × F1
def generate_Ck_Fk1_x_F1(F_prev, F1):
    Ck = set()

    # sorted list of items in F1
    items1 = sorted([list(i)[0] for i in F1])

    for f in F_prev:
        f_sorted = sorted(list(f))
        last = f_sorted[-1]

        for x in items1:
            if x > last:  # Apriori lexicographic constraint
                candidate = frozenset(f_sorted + [x])
                Ck.add(candidate)

    return Ck


# F(k-1) × F(k-1)
def generate_Ck_Fk1_x_Fk1(F_prev):
    Ck = set()

    # get the size of (k-1)
    any_itemset = next(iter(F_prev))
    k_minus_1 = len(any_itemset)

    # cas k = 2 : F_prev = F1 (singletons)
    # C2 = all pairs (i,j) with i < j and i,j ∈ F1
    if k_minus_1 == 1:
        items = sorted(next(iter(s)) for s in F_prev)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                Ck.add(frozenset([items[i], items[j]]))
        return Ck

    # case k ≥ 3 : standard Apriori self-join on prefixes
    L_list = sorted([sorted(list(itemset)) for itemset in F_prev])

    for i in range(len(L_list)):
        for j in range(i + 1, len(L_list)):
            I = L_list[i]
            J = L_list[j]

            # same prefix on the first k-2 elements
            if I[:-1] == J[:-1] and I[-1] < J[-1]:
                candidate = frozenset(I + [J[-1]])
                Ck.add(candidate)

    return Ck


def all_subsets_frequent(candidate, F_prev):
    k = len(candidate)
    for subset in itertools.combinations(candidate, k - 1):
        if frozenset(subset) not in F_prev:
            return False
    return True


def frequent_itemsets_gen(transactions, minsup, method="Fk-1xFk-1", verbose=False):
    # normalization
    transactions = normalize_transactions(transactions)
    N = len(transactions)

    # set of all items (for brute-force)
    all_items = set().union(*transactions)

    # C1 + F1
    C1 = generate_C1(transactions)
    support_C1 = compute_support(C1, transactions)
    F1 = {c for c, sup in support_C1.items() if sup >= minsup}

    if verbose:
        verbose_print_C1(C1, support_C1, minsup, N)

    all_frequent = set(F1)
    F_prev = F1
    k = 2

    while F_prev:
        # generation of Ck according to the method
        if method == "bruteforce":
            Ck = generate_Ck_bruteforce(all_items, k)
        elif method == "Fk-1xF1":
            Ck = generate_Ck_Fk1_x_F1(F_prev, F1)
        elif method == "Fk-1xFk-1":
            Ck = generate_Ck_Fk1_x_Fk1(F_prev)
        else:
            raise ValueError("Unknown method")

        # prune step : mark candidates not compatible with Apriori
        pruned = set()
        for c in list(Ck):
            if not all_subsets_frequent(c, F_prev):
                pruned.add(c)

        # we only count the support for the candidates not pruned
        Ck_no_prune = Ck - pruned

        # counting the support
        support_Ck = compute_support(Ck_no_prune, transactions)

        # Fk : frequent itemsets
        Fk = {c for c, sup in support_Ck.items() if sup >= minsup}

        if verbose:
            verbose_print_Ck(k, Ck, support_Ck, minsup, N, pruned)

        all_frequent |= Fk
        F_prev = Fk
        k += 1

    return all_frequent


# ---------- Format "style MLxtend" for the report ----------


def format_like_mlxtend(frequent_itemsets, transactions):
    formatted = []
    N = len(transactions)

    for itemset in frequent_itemsets:
        count = sum(itemset.issubset(t) for t in transactions)
        support = count / N
        formatted.append((support, tuple(sorted(itemset))))

    formatted.sort(key=lambda x: (len(x[1]), x[1]))

    return formatted


def print_mlxtend_style(formatted):
    print(f"{'support':<10} itemsets")
    for sup, items in formatted:
        if len(items) == 1:
            items_str = f"({items[0]})"
        else:
            items_str = "(" + ", ".join(items) + ")"
        print(f"{sup:<10} {items_str}")


if __name__ == "__main__":
    transactions = [
        ["Bread", "Milk"],
        ["Bread", "Diapers", "Beer", "Eggs"],
        ["Milk", "Diapers", "Beer", "Coke"],
        ["Bread", "Milk", "Diapers", "Beer"],
        ["Bread", "Milk", "Diapers", "Coke"],
    ]

    minsup = 0.6
    res = frequent_itemsets_gen(transactions, minsup, method="Fk-1xFk-1", verbose=True)
    formatted = format_like_mlxtend(res, transactions)
    print_mlxtend_style(formatted)
