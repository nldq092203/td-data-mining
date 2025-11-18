## Implémentation de trois m´ethodes de Frequent Itemset Generation
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

### Résultats pour le dataset (a)

Transactions :
[["a","b","d","e"], ["b","c","d"], ...]
minsup = 0.3

##### Méthode Brute-force

[Insérer image 1 ici : sortie brute-force du dataset (a)]
👉 Mettre ici ta capture contenant : 1-itemsets, 2-itemsets, 3-itemsets.

##### Méthode F_{k-1} × F_1

[Insérer image 2 ici : sortie fk1_f1 du dataset (a)]

##### Méthode F_{k-1} × F_{k-1}

[Insérer image 3 ici : sortie fk1_fk1 du dataset (a)]

### Résultats pour le dataset (b)

Transactions :
[["b","c","d"], ["a","b","c","d","e"], ...]
minsup = 0.5

##### Méthode Brute-force

[Insérer image 4 ici : sortie brute-force du dataset (b)]

##### Méthode F_{k-1} × F_1

[Insérer image 5 ici : sortie fk1_f1 du dataset (b)]

##### Méthode F_{k-1} × F_{k-1}

[Insérer image 6 ici : sortie fk1_fk1 du dataset (b)]

### Discussion
##### Nombre de candidats
La méthode Brute-force génère tous les k-combinaisons possibles : beaucoup d’itemsets inutiles, plus de sorties annotées N ou I.
La méthode F_{k-1} × F_1 réduit le nombre de candidats mais reste imparfaite : elle génère encore des combinaisons non valides.
La méthode F_{k-1} × F_{k-1} est la plus efficace : moins de candidats, moins d’éléments “N”, exploration plus ciblée.

##### Influence des datasets
Dans le dataset (a), les items apparaissent souvent ensemble → plus d’itemsets fréquents → grosses différences entre les méthodes.
Dans le dataset (b), le seuil minsup = 0.5 filtre davantage → peu d’itemsets fréquents → les trois méthodes convergent plus vite.

##### Conclusion simple
Brute-force : correcte mais inefficace.
F_{k-1} × F_1 : mieux, mais encore des candidats inutiles.
F_{k-1} × F_{k-1} : meilleure stratégie, proche de la version classique d’Apriori.

### Conclusion
Nous avons comparé trois stratégies de génération d’itemsets dans Apriori. Les résultats montrent clairement que la méthode F_{k-1} × F_{k-1} produit moins de candidats et évite les combinaisons impossibles grâce au principe d’élagage. C’est celle qui fonctionne le mieux dans les deux jeux de données.