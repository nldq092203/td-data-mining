import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle


def euclidean(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))


def plot_clusters(data, centroids, clusters, names=None, iteration=None):
    colors = ["r", "g", "b", "c", "m", "y"]
    plt.figure()
    for i, cluster in enumerate(clusters):
        if cluster:
            xs, ys = zip(*cluster)
            plt.scatter(xs, ys, color=colors[i], label=f"Cluster {i+1}")
            plt.scatter(
                centroids[i][0],
                centroids[i][1],
                marker=MarkerStyle("x"),
                s=200,
                color=colors[i],
                edgecolor="k",
            )
    if names:
        for i, (x, y) in enumerate(data):
            plt.text(x + 0.1, y + 0.1, names[i])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Iteration {iteration}" if iteration else "Final clustering")
    plt.grid(True)
    plt.legend()
    plt.show()


def kmeans_2d(data, k, centroids_init=None, names=None, max_iter=10, visualize=True, verbose=True):
    if centroids_init is None:
        centroids = random.sample(data, k)
    else:
        centroids = [list(c) for c in centroids_init]

    for it in range(1, max_iter + 1):
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [euclidean(point, c) for c in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)

        if verbose:
            print(f"\nIteration {it}")
            for i, cluster in enumerate(clusters):
                print(f"Centroid {centroids[i]} Points {cluster}")

        if visualize:
            plot_clusters(data, centroids, clusters, names, iteration=it)

        new_centroids = []
        for cluster in clusters:
            if cluster:
                mean_point = np.mean(cluster, axis=0)
                new_centroids.append([float(x) for x in mean_point])
            else:
                new_centroids.append(random.choice(data))

        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    sse = sum(
        euclidean(p, centroids[i]) ** 2 for i, c in enumerate(clusters) for p in c
    )
    print(f"Fin clustering, erreur = {sse:.2f}")
    return centroids, clusters, sse


if __name__ == "__main__":
    points = [[-2, 3], [-2, 1], [-2, -1], [2, -1], [2, 1], [1, 0]]
    names = ["M1", "M2", "M3", "M4", "M5", "M6"]

    print("=== Cas (a): M1 et M2 ===")
    kmeans_2d(points, k=2, centroids_init=[[-2, 3], [-2, 1]], names=names, visualize=True)

    print("=== Cas (b): M4 et M6 ===")
    kmeans_2d(points, k=2, centroids_init=[[2, -1], [1, 0]], names=names, visualize=True)
