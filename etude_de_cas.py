import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from numpy import array, identity, transpose, matmul, std, mean
from numpy.linalg import eig
from scipy.cluster.hierarchy import linkage, dendrogram

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from k_means_2D import kmeans_2d

def parse_country_data(filename):
    data_country = []
    country_names = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            country_names.append(row['country'])
            
            country_data = [
                float(row['child_mort']),
                float(row['exports']),
                float(row['health']),
                float(row['imports']),
                float(row['income']),
                float(row['inflation']),
                float(row['life_expec']),
                float(row['total_fer']),
                float(row['gdpp'])
            ]
            
            data_country.append(country_data)
    
    return data_country, country_names


def pca_reduction(data_country, countries):
    # Les mêmes données de la matrice data_country, dans le format "array" exploitable pour la suite
    X = array(data_country)
    
    # Le nombre de points
    n = len(countries)
    
    # Le nombre de variables
    p = len(data_country[0])
    
    # La matrice des données centrées, c'est-à-dire la somme = 0 sur chaque colonne
    Y = X - matmul(transpose(array([n * [1]])), transpose(array([[mean(variable)] for variable in transpose(X)])))
    
    # La matrice des données centrées et réduites, qui en plus a l'écart type constant = 1 sur chaque colonne
    Z = matmul(Y, array(list(map(lambda variable: [1./std(variable)], transpose(X)))) * identity(p))
    
    # La matrice (symétrique) de variance/covariance des données centrées réduites
    R = matmul(matmul(transpose(Z), 1./n * identity(n)), Z)
    
    # Les vecteurs propres de R
    eigenvectors = eig(R)[1]
    
    # Les 2 composantes principales = 2 nouvelles variables contenant le plus d'information possible des 9 variables initiales
    components = [matmul(Z, eigenvectors[:,0]), matmul(Z, eigenvectors[:,1])]
    
    # La matrice initiale, projetée sur 2 nouvelles colonnes qui représentent les 2 composantes principales
    data_reduced = [[components[0][i], components[1][i]] for i in range(n)]
    
    return data_reduced


def plot_kmeans_with_labels(data, centroids, clusters, country_names, k, filename=None):
    colors = ["r", "g", "b", "c", "m", "y", "orange", "purple", "brown", "pink"]
    plt.figure(figsize=(16, 12))
    
    # Créer un mapping point -> nom de pays
    point_to_name = {tuple(data[i]): country_names[i] for i in range(len(data))}
    
    for i, cluster in enumerate(clusters):
        if cluster:
            xs, ys = zip(*cluster)
            plt.scatter(xs, ys, color=colors[i % len(colors)], alpha=0.6, 
                       label=f"Cluster {i+1} ({len(cluster)} pays)", s=50)
            
            for point in cluster:
                country = point_to_name[tuple(point)]
                plt.annotate(country, (point[0], point[1]), 
                           fontsize=6, alpha=0.7, 
                           xytext=(3, 3), textcoords='offset points')
            
            plt.scatter(
                centroids[i][0],
                centroids[i][1],
                marker='x',
                s=300,
                color=colors[i % len(colors)],
                edgecolor='k',
                linewidths=3,
                label=f"Centroïde {i+1}"
            )
    
    plt.xlabel("Première composante principale", fontsize=12)
    plt.ylabel("Deuxième composante principale", fontsize=12)
    plt.title(f"Clustering K-means avec k={k} (avec labels)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée : {filename}")
    else:
        plt.show()
    
    plt.close()


def hierarchical_clustering(data, country_names, method='single', filename=None):
    Z = linkage(data, method=method)
    
    plt.figure(figsize=(20, 10))
    dendrogram(Z, labels=country_names, leaf_font_size=7)
    
    method_name = "Simple (Single Link)" if method == 'single' else "Complète (Complete Link)"
    plt.title(f"Clustering Hiérarchique - Liaison {method_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Pays", fontsize=12)
    plt.ylabel("Distance", fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée : {filename}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Lecture des données depuis le fichier Country-data.csv
    data_country, country_names = parse_country_data('./Country-data.csv')
    countries = country_names  # Alias pour la compatibilité avec le code fourni
    
    print(f"\nFirst country: {country_names[0]}")
    print(f"Data: {data_country[0]}")
    print(f"\nSecond country: {country_names[1]}")
    print(f"Data: {data_country[1]}")
    
    # Appel de la fonction PCA
    data_reduced = pca_reduction(data_country, countries)
    
    print(f"\ndata_reduced commence par: [[{data_reduced[0][0]:.2f}, {data_reduced[0][1]:.2f}], [{data_reduced[1][0]:.2f}, {data_reduced[1][1]:.2f}], ...]")
    
    centroids_2, clusters_2, sse_2 = kmeans_2d(
        data_reduced, k=2, names=None, max_iter=100, visualize=False, verbose=False
    )
    
    # Visualisation finale avec labels et sauvegarde
    plot_kmeans_with_labels(data_reduced, centroids_2, clusters_2, country_names, 
                            k=2, filename='country_data_k2.png')
    
    # K-means avec k=3
    centroids_3, clusters_3, sse_3 = kmeans_2d(
        data_reduced, k=3, names=None, max_iter=100, visualize=False, verbose=False
    )
    
    # Visualisation finale avec labels et sauvegarde
    plot_kmeans_with_labels(data_reduced, centroids_3, clusters_3, country_names, 
                            k=3, filename='country_data_k3.png')

    # Clustering hiérarchique - Single Link
    hierarchical_clustering(data_reduced, country_names, method='single', 
                           filename='country_data_single_link.png')
    
    # Clustering hiérarchique - Complete Link
    hierarchical_clustering(data_reduced, country_names, method='complete',
                           filename='country_data_complete_link.png')