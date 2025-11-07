## 1 Clustering k-means
### 1.1 Implémentation
Le fonction `kmeans(data, k, centroids_init=None, max_iter=100)` qui applique l’algorithme k-means dans le fichier *k_means_1D.py*
### 1.2 Tests sur le jeu de données 1D
Jeu de données :  
`[[2], [4], [6], [12], [24], [30]]`
(a) Avec centroïdes initiaux  `[[2], [6]]`
Résultat:

```
Iteration 1
Centroid [2] Points [[2], [4]]
Centroid [6] Points [[6], [12], [24], [30]]

Iteration 2
Centroid [3.0] Points [[2], [4], [6]]
Centroid [18.0] Points [[12], [24], [30]]

Iteration 3
Centroid [4.0] Points [[2], [4], [6], [12]]
Centroid [22.0] Points [[24], [30]]

Iteration 4
Centroid [6.0] Points [[2], [4], [6], [12]]
Centroid [27.0] Points [[24], [30]]
Fin clustering, erreur = 74.00
```

(b) Avec centroïdes initiaux `[[12], [24]]`
Résultat:

```
Iteration 1
Centroid [12] Points [[2], [4], [6], [12]]
Centroid [24] Points [[24], [30]]

Iteration 2
Centroid [6.0] Points [[2], [4], [6], [12]]
Centroid [27.0] Points [[24], [30]]
Fin clustering, erreur = 74.00
```

Les deux configurations de centroïdes initiaux aboutissent à la même partition finale :
- Cluster 1 : `[[2], [4], [6], [12]]`
- Cluster 2:  `[[24], [30]]`
- Centroïdes finaux : `[6.0] et [27.0]`
- Erreur (SSE) = 74.00
L’algorithme converge vers la même solution optimale, quel que soit le choix initial des centroïdes.  
Aucun clustering n’est meilleur que l’autre : les deux donnent la même erreur et la même séparation logique des données.


### 1.3 Visualisation 2D
La fonction `kmeans_2d` dans le fichier *k_means_2D.py* , version étendue de `kmeans`, affiche la répartition des points et des centroïdes à chaque itération à l’aide de *matplotlib*.
Paramètres optionnels ajoutés :
- `names` : liste des noms des points, utilisée pour annoter chaque point sur la figure.
- `visualize` : booléen (True par défaut) activant la génération automatique des figures à chaque itération.

Jeu de données :  
M1(−2, 3), M2(−2, 1), M3(−2, −1), M4(2, −1), M5(2, 1), M6(1, 0)

(a) Avec centroïdes initiaux M1 et M2:
![[Pasted image 20251107064426.png|320x300]] ![[Pasted image 20251107064700.png|320x300]]
![[Pasted image 20251107064822.png|320x300]]


```
points=[[-2,3],[-2,1],[-2,-1],[2,-1],[2,1],[1,0]]
names=["M1","M2","M3","M4","M5","M6"]
kmeans_2d(points,2,centroids_init=[[-2,3],[-2,1]],names=names)
```
Résultat:
```
Iteration 1
Centroid [-2, 3] Points [[-2, 3]]
Centroid [-2, 1] Points [[-2, 1], [-2, -1], [2, -1], [2, 1], [1, 0]]

Iteration 2
Centroid [-2.0, 3.0] Points [[-2, 3], [-2, 1]]
Centroid [0.2, 0.0] Points [[-2, -1], [2, -1], [2, 1], [1, 0]]

Iteration 3
Centroid [-2.0, 2.0] Points [[-2, 3], [-2, 1]]
Centroid [0.75, -0.25] Points [[-2, -1], [2, -1], [2, 1], [1, 0]]
Fin clustering, erreur = 15.50
```
Après 3 itérations, les points se regroupent en deux clusters stables :
- **C1 (rouge)** : M1, M2
- **C2 (vert)** : M3, M4, M5, M6
Les centroïdes finaux `[-2.0, 2.0]` et `[0.75, -0.25]`

(b) Avec centroïdes initiaux M4 et M6:

![[Pasted image 20251107065343.png|320x300]] ![[Pasted image 20251107065441.png|320x300]]
![[Pasted image 20251107065519.png|320x300]]
```
points=[[-2,3],[-2,1],[-2,-1],[2,-1],[2,1],[1,0]]
names=["M1","M2","M3","M4","M5","M6"]
kmeans_2d(points,2,centroids_init=[[2,-1],[1,0]],names=names)
```
Résultat:
```
Iteration 1
Centroid [2, -1] Points [[2, -1]]
Centroid [1, 0] Points [[-2, 3], [-2, 1], [-2, -1], [2, 1], [1, 0]]

Iteration 2
Centroid [2.0, -1.0] Points [[2, -1], [2, 1], [1, 0]]
Centroid [-0.6, 0.8] Points [[-2, 3], [-2, 1], [-2, -1]]

Iteration 3
Centroid [1.6666666666666667, 0.0] Points [[2, -1], [2, 1], [1, 0]]
Centroid [-2.0, 1.0] Points [[-2, 3], [-2, 1], [-2, -1]]
Fin clustering, erreur = 10.67
```
Après 3 itérations, les points se regroupent en deux clusters stables :
- C1 (rouge)** : M4, M5, M6
- **C2 (vert)** : M1, M2, M3
Les centroïdes finaux `[1.67, 0.0]` et `[-2.0, 1.0]`

**Interprétation finale** — cas (a) & (b)
- Les deux initialisations convergent en 3 itérations vers 2 clusters stables, mais l’affectation de M3 dépend de l’initialisation.
- **Cas (a)** (centroïdes finaux `[-2,2]` et `[0.75,-0.25]`) : clusters {M1,M2} et {M3,M4,M5,M6} avec SSE = 15.50.
- **Cas (b)** (centroïdes finaux `[1.67,0]` et `[-2,1]`) : clusters {M4,M5,M6} et {M1,M2,M3} avec SSE = 10.67
**Conclusion**: Le cas (b) donne une séparation plus naturelle et une erreur (SSE) plus faible, donc un clustering de meilleure qualité.

### 1.4 Vérification avec Scikit-Learn
(a) Jeu de données `[[1],[2],[18],[20],[31]]`
Les valeurs d’étiquettes (0/1/2) sont arbitraires :
- première init : résultat = `[0 1 2 2 2]` alors clusters = `{[1]}, {[2]}, {[18],[20],[31]}` ;
- seconde init : résultat = `[0 0 1 1 2]` alors clusters = `{[1],[2]}, {[18],[20]}, {[31]}` ;
Ce sont exactement les mêmes partitions que dans 1.1 (à permutation près des labels).

(b) Exercice 2 - `[[2],[4],[6],[12],[24],[30]]`, `k=2`

```
data2 = array([[2], [4], [6], [12], [24], [30]])

print(KMeans(n_clusters=2, n_init=1, init=array([[2], [6]])).fit(data2).labels_)
print(KMeans(n_clusters=2, n_init=1, init=array([[12], [24]])).fit(data2).labels_)
```
Résultat:
```
[0 0 0 0 1 1]
[0 0 0 0 1 1]
```
Dans les deux cas, mêmes clusters finaux que notre code : {[2],[4],[6],[12]} et {[24],[30]}.

(c) Exercice 3 -  `M1(−2, 3), M2(−2, 1), M3(−2, −1), M4(2, −1), M5(2, 1), M6(1, 0)` `k=3`
```
data3 = array([[-2, 3], [-2, 1], [-2, -1], [2, -1], [2, 1], [1, 0]])

# Cas (a) : init M1 & M2
print(KMeans(n_clusters=2, n_init=1, init=array([[-2, 3], [-2, 1]])).fit(data3).labels_)

# Cas (b) : init M4 & M6
print(KMeans(n_clusters=2, n_init=1, init=array([[2, -1], [1, 0]])).fit(data3).labels_)
```

Résultat:

```
[0 0 1 1 1 1]
[1 1 1 0 0 0]
```

Cas (a) : clusters {M1,M2} et {M3,M4,M5,M6} (comme notre implémentation).
Cas (b) : clusters {M1,M2,M3} et {M4,M5,M6} (comme notre implémentation).

**Conclusion** — Pour 1.1, 1.2 et 1.3, sklearn.KMeans retrouve les mêmes partitions que notre code (à permutation des labels près), validant l’implémentation.