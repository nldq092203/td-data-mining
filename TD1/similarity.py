import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

similarity = np.array(
    [
        [1.00, 0.10, 0.41, 0.55, 0.35],
        [0.10, 1.00, 0.64, 0.47, 0.98],
        [0.41, 0.64, 1.00, 0.44, 0.85],
        [0.55, 0.47, 0.44, 1.00, 0.76],
        [0.35, 0.98, 0.85, 0.76, 1.00],
    ]
)

# Conversion en matrice de distance
distance = 1 - similarity

data = squareform(distance)
labels = ["p1", "p2", "p3", "p4", "p5"]

# Single link
Z_single = linkage(data, method="single")
plt.figure()
dendrogram(Z_single, labels=labels)
plt.title("Single link (à partir de similarité)")
plt.show()

# Complete link
Z_complete = linkage(data, method="complete")
plt.figure()
dendrogram(Z_complete, labels=labels)
plt.title("Complete link (à partir de similarité)")
plt.show()
