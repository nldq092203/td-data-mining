import random
import numpy as np


def euclidean(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))


def kmeans(data, k, centroids_init=None, max_iter=100):
    if centroids_init is None:
        centroids = random.sample(data, k)
    else:
        centroids = [list(c) for c in centroids_init]

    for it in range(max_iter):
        clusters = [[] for _ in range(k)]

        for point in data:
            distances = [euclidean(point, c) for c in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)

        print(f"\nIteration {it+1}")
        for i, cluster in enumerate(clusters):
            print(f"Centroid {centroids[i]} Points {cluster}")

        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroids.append([float(x) for x in np.mean(cluster, axis=0)])
            else:
                new_centroids.append(random.choice(data))

        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    # Compute SSE
    sse = 0
    for i, cluster in enumerate(clusters):
        for point in cluster:
            sse += euclidean(point, centroids[i]) ** 2

    print(f"Fin clustering, erreur = {sse:.2f}")
    return centroids, clusters, sse


data = [[2], [4], [6], [12], [24], [30]]
kmeans(data, k=2, centroids_init=[[2], [6]])
kmeans(data, k=2, centroids_init=[[12], [24]])
