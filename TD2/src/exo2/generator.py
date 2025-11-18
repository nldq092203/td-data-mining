import random


def generate_transactions(
    n_transactions: int,
    n_items: int,
    width_range: tuple[int, int] = (5, 20),
    popular_ratio: float = 0.0,
    popular_weight: float = 5.0,
):
    """
    Génère un dataset synthétique de transactions.
    
    Paramètres :
        - n_transactions : nombre total de transactions
        - n_items        : nombre d'items p0, p1, ..., p(n_items-1)
        - width_range    : intervalle du nombre d'items par transaction (min, max)
        - popular_ratio  : fraction des items considérés comme "populaires".
                           Lorsque > 0, ces items sont sélectionnés avec un poids
                           plus élevé pour créer un dataset déséquilibré et mettre
                           en évidence les différences entre les méthodes Apriori.
        - popular_weight : poids relatif des items populaires par rapport aux autres.

    Retour :
        list[list[str]] : liste de transactions
    """
    all_items = [f"p{i}" for i in range(n_items)]
    transactions = []

    use_weights = 0 < popular_ratio < 1 and n_items > 0
    if use_weights:
        n_popular = max(1, int(popular_ratio * n_items))
        popular_set = set(all_items[:n_popular])
        weights = [
            popular_weight if item in popular_set else 1.0 for item in all_items
        ]

    for _ in range(n_transactions):
        k = min(random.randint(*width_range), n_items)

        if use_weights:
            picked = random.choices(all_items, weights=weights, k=k)
            trans = list(dict.fromkeys(picked))

            if len(trans) < k:
                remaining = [item for item in all_items if item not in trans]
                extra = random.sample(remaining, k - len(trans))
                trans.extend(extra)
        else:
            trans = random.sample(all_items, k)

        transactions.append(trans)

    return transactions
