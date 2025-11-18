from itertools import combinations
from collections import defaultdict

def count_support(transactions, candidates):
    """
    Count how many transactions contain each candidate itemset.
    
    transactions : list of lists
    candidates   : list of tuples
    return       : dict {(itemset tuple): count}
    """
    support = defaultdict(int)

    for t in transactions:
        tset = set(t)
        for c in candidates:
            if set(c).issubset(tset):
                support[c] += 1

    return dict(support)

def split_frequent(support_dict, minsup, n_transactions):
    """
    support_dict : {itemset: count}
    minsup       : float (0.6)
    n_transactions : int

    return:
      - frequent : list[itemsets]
      - status   : dict {itemset: ("F" or "I", count)}
    """
    frequent = []
    status = {}
    min_cnt = minsup * n_transactions

    for itemset, cnt in support_dict.items():
        if cnt >= min_cnt:
            frequent.append(itemset)
            status[itemset] = ("F", cnt)
        else:
            status[itemset] = ("I", cnt)

    return frequent, status

def gen_bruteforce(k, all_items):
    """Generate all k-combinations of all items."""
    return [tuple(sorted(c)) for c in combinations(all_items, k)]

def gen_fk1_f1(F_prev, F1):
    """
    Join each frequent (k-1)-itemset with each frequent 1-itemset.
    """
    candidates = set()

    for itemset in F_prev:
        for (i,) in F1:
            if i not in itemset:
                new = tuple(sorted(itemset + (i,)))
                if len(new) == len(itemset) + 1:
                    candidates.add(new)

    return sorted(candidates)

def gen_fk1_fk1(F_prev):
    """
    Self-join: two (k-1)-itemsets must share their first k-2 items.
    """
    candidates = set()
    F_sorted = sorted(F_prev)

    for i in range(len(F_sorted)):
        for j in range(i + 1, len(F_sorted)):
            a, b = F_sorted[i], F_sorted[j]

            # Join condition: common prefix of length k-2
            if a[:-1] == b[:-1]:
                new = tuple(sorted(set(a) | set(b)))
                candidates.add(new)

    return sorted(candidates)

def apriori_prune(Ck, F_prev, k):
    """
    Remove candidates that contain an infrequent (k-1)-subset.
    
    Return:
      - to_count : list (need support computation)
      - pruned   : list (inferred infrequent → label "N")
    """
    F_prev_set = set(F_prev)
    to_count, pruned = [], []

    for c in Ck:
        ok = True
        for sub in combinations(c, k - 1):
            if tuple(sorted(sub)) not in F_prev_set:
                ok = False
                break

        if ok:
            to_count.append(c)
        else:
            pruned.append(c)

    return to_count, pruned

def apriori_frequent_itemsets(
    transactions,
    minsup,
    method="bruteforce",
    verbose=False,
    prune=True,
):
    """
    Implements the Frequent Itemset Generation phase of Apriori.

    Parameters:
        transactions : dataset as list[list[str]]
        minsup       : minimum support (0..1)
        method       : "bruteforce", "fk1_f1", or "fk1_fk1"
        verbose      : print intermediate information

    Returns:
        all_F : dict {k: list of frequent k-itemsets}
    """
    n_transactions = len(transactions)

    # 1 - itemsets
    all_items = sorted({item for t in transactions for item in t})
    C1 = [(i,) for i in all_items]

    support1 = count_support(transactions, C1)
    F1, status1 = split_frequent(support1, minsup, n_transactions)

    if verbose:
        print("=" * 30)
        pretty_print_itemsets(
            "1-itemsets",
            C1,
            status_dict=status1,
            pruned=None
        )

    all_F = {1: F1}
    F_prev = F1
    k = 2

    # K-itemsets (k ≥ 2)
    while F_prev:
        # Candidate generation
        if method == "bruteforce":
            Ck = gen_bruteforce(k, all_items)
        elif method == "fk1_f1":
            Ck = gen_fk1_f1(F_prev, F1)
        elif method == "fk1_fk1":
            Ck = gen_fk1_fk1(F_prev)

        # Prune (produce N)
        to_count, pruned = apriori_prune(Ck, F_prev, k)

        # Count support only for survivors
        support_k = count_support(transactions, to_count)
        Fk, status_k = split_frequent(support_k, minsup, n_transactions)

        if verbose:
            print("\n" + "=" * 30)
            pretty_print_itemsets(
                f"{k}-itemsets",
                to_count,
                status_dict=status_k,
                pruned=pruned
            )

        if not Fk:
            break

        all_F[k] = Fk
        F_prev = Fk
        k += 1
    
    return all_F

def pretty_print_itemsets(title, candidates, status_dict=None, pruned=None):
    """
    Print itemsets in a clean, aligned, typewriter-style format.

    title        : string label ("1-itemsets")
    candidates   : list of itemsets to print (tuples)
    status_dict  : dict {itemset: ("F"/"I", count)} (None when printing only N)
    pruned       : list of pruned candidates (label "N")
    """
    print(title)
    
    item_width = 20
    
    if pruned:
        for it in sorted(pruned):
            txt = ",".join(it)
            print(f"{txt:<{item_width}} N")

    if status_dict:
        for it in candidates:
            label, cnt = status_dict[it]
            txt = ",".join(it)
            print(f"{txt:<{item_width}} {cnt:<2} {label}")


if __name__ == "__main__":
    transactions = [
        ["Bread", "Milk"],
        ["Bread", "Diapers", "Beer", "Eggs"],
        ["Milk", "Diapers", "Beer", "Coke"],
        ["Bread", "Milk", "Diapers", "Beer"],
        ["Bread", "Milk", "Diapers", "Coke"],
    ]

    print("Méthode Brute-force")
    apriori_frequent_itemsets(transactions, 0.6, "bruteforce", True)

    print("\nMéthode F_{k-1} x F_1")
    apriori_frequent_itemsets(transactions, 0.6, "fk1_f1", True)

    print("\nMéthode F_{k-1} x F_{k-1}")
    apriori_frequent_itemsets(transactions, 0.6, "fk1_fk1", True)
