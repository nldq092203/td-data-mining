## Implémentation de trois méthodes de Frequent Itemset Generation
### Introduction

Nous avons implémenté la phase Frequent Itemset Generation de l’algorithme Apriori.
Trois méthodes de génération des candidats ont été testées :
- Méthode 1 : Brute-force
- Méthode 2 : F_{k-1} × F_1
- Méthode 3 : F_{k-1} × F_{k-1}
Chaque itemset produit par notre fonction est annoté :
- F : itemset fréquent
- I : itemset infrequent (après calcul du support)
- N : itemset éliminé directement par le principe Apriori
Nous présentons les résultats obtenus pour les jeux de données (a) et (b), suivis d’une comparaison simple entre les trois méthodes.

### Résultats pour le dataset example
Pour obtenir les résultats pour le dataset exemple, exécutez :
```
python .\src\exo1\apriori.py
```

##### Méthode Brute-force

![dataset exemple - brute-force](/static/exo1/image1.png)

##### Méthode F_{k-1} × F_1

![dataset exemple - Fk1 x F1](/static/exo1/image2.png)

##### Méthode F_{k-1} × F_{k-1}

![dataset exemple - Fk1 x Fk1](/static/exo1/image3.png)

### Résultats pour le dataset (a)

Pour obtenir les résultats pour le dataset (a) and (b), exécutez :
```
python .\src\exo1\test_apriori.py   
```

Transactions :
[["a", "b", "d", "e"], ["b", "c", "d"], ["a", "b", "d", "e"], 
["a", "c", "d","e"], ["b", "c", "d", "e"], ["b", "d", "e"], 
["c", "d"], ["a", "b", "c"], ["a", "d", "e"],["b", "d"]]
minsup = 0.3

##### Méthode Brute-force

![dataset (a) - brute-force](/static/exo1/image4.png)

##### Méthode F_{k-1} × F_1

![dataset (a) - Fk1 x F1](/static/exo1/image5.png)

##### Méthode F_{k-1} × F_{k-1}

![dataset (a) - Fk1 x Fk1](/static/exo1/image6.png)

### Résultats pour le dataset (b)

Transactions :
[["b", "c", "d"], ["a", "b", "c", "d", "e"], 
["a", "b", "c", "e"], ["a", "b", "d", "e"], 
["b", "c", "e"], ["a", "b", "d", "e"]]
minsup = 0.5

##### Méthode Brute-force

![dataset (b) - brute-force](/static/exo1/image7.png)

##### Méthode F_{k-1} × F_1

![dataset (b) - Fk1 x F1](/static/exo1/image8.png)

##### Méthode F_{k-1} × F_{k-1}

![dataset (b) - Fk1 x Fk1](/static/exo1/image9.png)

### Discussion
##### Nombre de candidats
La méthode Brute-force génère tous les k-combinaisons possibles : beaucoup d'itemsets inutiles, plus de sorties annotées N ou I. Par exemple, dans le dataset (a) à k=3, elle génère 10 candidats dont 5 sont marqués "N".
La méthode F_{k-1} × F_1 réduit le nombre de candidats mais reste imparfaite : elle génère encore des combinaisons non valides. Dans le dataset (a), à k=3 elle génère aussi 10 candidats (5 N), mais à k=4 elle réduit à 3 candidats (tous N) contre 5 pour Brute-force.
La méthode F_{k-1} × F_{k-1} est la plus efficace : moins de candidats, moins d'éléments "N", exploration plus ciblée. Dans le dataset (a), à k=3 elle génère seulement 6 candidats (1 N) au lieu de 10, et à k=4 elle génère 0 candidat car l'élagage élimine tout.

##### Influence des datasets
Dans le dataset (a), les items apparaissent souvent ensemble -> plus d'itemsets fréquents -> différences visibles entre les méthodes (notamment à k=4 : Brute-force génère 5 candidats, F_{k-1}xF_1 en génère 3).
Dans le dataset (b), le seuil minsup = 0.5 filtre davantage -> Brute-force et F_{k-1}xF_1 génèrent exactement les mêmes candidats, tandis que F_{k-1}xF_{k-1} reste la plus efficace avec significativement moins de candidats.

### Conclusion
Nous avons comparé trois stratégies de génération d'itemsets dans Apriori. Les résultats montrent clairement que :
- **Brute-force** : correcte mais inefficace, génère tous les k-combinaisons possibles.
- **F_{k-1} × F_1** : amélioration notable, mais génère encore des candidats inutiles.
- **F_{k-1} × F_{k-1}** : meilleure stratégie, proche de la version classique d'Apriori. Elle produit significativement moins de candidats et évite les combinaisons impossibles grâce au principe d'élagage. C'est celle qui fonctionne le mieux dans les deux jeux de données.

## Comparaison des implémentations de la Section 1

### Exécution du benchmark

Pour exécuter le benchmark et générer les deux figures de comparaison :

```bash
python .\src\exo2\benchmark.py
```

Ce script va :
1. Exécuter le premier benchmark avec des paramètres faciles (20 items, transactions courtes, minsup=0.4)
2. Générer et sauvegarder la Figure 1 dans `static/exo2/figure1_facile.png`
3. Exécuter le second benchmark avec des paramètres difficiles (200 items, transactions larges, minsup=0.25)
4. Générer et sauvegarder la Figure 2 dans `static/exo2/figure2_difficile.png`

Chaque benchmark teste les trois méthodes (Brute-force, F_{k-1} × F_1, F_{k-1} × F_{k-1}) sur trois tailles de datasets (1000, 2000, 3000 transactions) et moyenne les résultats sur 3 exécutions.

### Générateur synthétique de transactions
Pour cette partie, nous avons d'abord essayé un générateur totalement aléatoire :

- on choisit n_items items notés p0, p1, ..., p(n_items-1)
- pour chaque transaction, on tire au hasard un nombre d'items dans width_range
- on prend un échantillon uniforme parmi tous les items

Ce générateur fonctionne, mais dans la pratique les trois méthodes ont presque toujours le même temps :

- la répartition des items est trop uniforme

- les grands itemsets fréquents sont rares

- le nombre de candidats à chaque niveau reste limité

- avec le pruning d’Apriori, les trois méthodes ont quasiment la même charge de travail


**Résultat :** les courbes de temps sont superposées, donc on ne voit pas bien la différence entre Brute-force et les deux autres méthodes.

Pour obtenir des comportements plus intéressants, nous avons ensuite utilisé un générateur déséquilibré, avec des items populaires :
```
generate_transactions(
    n_transactions,
    n_items,
    width_range=(min_k, max_k),
    popular_ratio=...,
    popular_weight=...,
)
```
Idée :

- une petite fraction des items (popular_ratio, par ex. 15 %) est marquée comme populaire

- ces items sont tirés avec un poids plus élevé (popular_weight, par ex. 12.0)

- ils apparaissent donc très souvent et ensemble dans les transactions

Effet :

- les items populaires créent beaucoup plus de combinaisons fréquentes

- cela augmente fortement le nombre de candidats pour les méthodes naïves

- la méthode Brute-force devient nettement plus lente que les deux versions avec joins basées sur F_{k-1}

### Première figure : cas où les méthodes sont similaires
Pour la première figure, nous avons choisi des paramètres faciles :

- n_items = 20

- width_range = (2, 5)

- minsup = 0.4

- sizes = [1000, 2000, 3000]

- popular_ratio = 0.15

- popular_weight = 12.0

- n_runs = 3 (moyenne sur 3 exécutions pour lisser les temps)

Dans ce scénario :

- peu d’items

- transactions assez courtes

- seuil minsup relativement élevé

Le nombre de candidats reste modéré, même si certains items sont populaires.

**Résultat :** les trois méthodes (Brute-force, F_{k-1} × F_1 et F_{k-1} × F_{k-1}) ont des temps très proches.

##### Figure 1 : Paramètres faciles

![Comparaison avec paramètres faciles](/static/exo2/figure1_facile.png)

Sur cette figure, on voit que :

- les trois courbes sont presque superposées

- le choix de la méthode de génération de candidats n'a pas beaucoup d'impact sur le temps total

### Deuxième figure : cas où Brute-force devient beaucoup plus lente
Pour la deuxième figure, nous avons choisi des paramètres difficiles :

- n_items = 200
- width_range = (25, 30)
- minsup = 0.25
- sizes = [1000, 2000, 3000]
- même réglage d'items populaires (popular_ratio = 0.15, popular_weight = 12.0)
- n_runs = 3

Dans ce cas :

- beaucoup plus d'items disponibles
- transactions très larges (25–30 items chacune)
- seuil minsup plus bas

Les items populaires se retrouvent dans un grand nombre de transactions.

Cela crée :

- de très nombreux itemsets fréquents
- une explosion du nombre de candidats possibles pour les niveaux k ≥ 2

La méthode Brute-force génère tous les candidats possibles (toutes les k-combinaisons), ce qui représente un très grand nombre initial de candidats. Bien que le pruning d'Apriori soit ensuite appliqué à tous les candidats, cette méthode :

- doit d'abord générer un nombre énorme de candidats (C(n,k))
- doit ensuite traiter tous ces candidats lors de l'étape de pruning
- devient rapidement beaucoup plus lente que les deux méthodes basées sur F_{k-1}

En revanche, les méthodes F_{k-1} × F_1 et F_{k-1} × F_{k-1} génèrent intelligemment un ensemble de candidats beaucoup plus restreint dès le départ. Même si le pruning est ensuite appliqué, elles partent avec beaucoup moins de candidats à traiter. Elles restent donc relativement efficaces.

##### Figure 2 : Paramètres difficiles

![Comparaison avec paramètres difficiles](/static/exo2/figure2_difficile.png)

Sur cette figure, on observe clairement :

- la courbe Brute-force qui croît beaucoup plus vite
- les courbes F_{k-1} × F_1 et F_{k-1} × F_{k-1} qui restent proches et nettement en dessous

### Discussion

Un générateur complètement aléatoire donne souvent des datasets trop uniformes :
dans ce cas, la structure des candidats est similaire pour les trois méthodes et leurs temps d'exécution sont presque identiques.

Le générateur avec items populaires permet de créer des scénarios plus réalistes :
certains items reviennent souvent, les combinaisons explosent pour Brute-force, alors que les méthodes F_{k-1} × F_1 et F_{k-1} × F_{k-1} exploitent mieux l'information des itemsets fréquents précédents.

**Récapitulatif des résultats attendus :**

- **Figure 1** (paramètres faciles) : les trois méthodes se comportent de manière similaire, courbes superposées.

- **Figure 2** (paramètres difficiles) : Brute-force devient clairement la pire méthode avec un temps d'exécution qui croît beaucoup plus rapidement.

### Conclusion
Cet exercice montre que pour comparer correctement les implémentations d'Apriori, le choix du générateur de données et des paramètres est crucial :

- **Générateur naïf** : les différences de performance restent cachées car la distribution uniforme des items limite le nombre de candidats.

- **Générateur déséquilibré** : avec des items populaires et des paramètres adaptés, la méthode Brute-force ne passe pas à l'échelle, alors que les méthodes basées sur F_{k-1} sont beaucoup plus robustes grâce à l'exploitation du principe d'Apriori.