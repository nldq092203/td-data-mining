from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

data = [0.1, 0.9, 0.35, 0.8, 0.3, 0.4, 0.5, 0.6, 0.7, 0.2]
labels = ["p1", "p2", "p3", "p4", "p5"]

# --- Clustering hiérarchique avec liaison simple ---
Z_single = linkage(data, method="single")
plt.figure(figsize=(6, 4))
dendrogram(Z_single, labels=labels)
plt.title("Clustering hiérarchique - Liaison simple (Single Link)")
plt.xlabel("Points")
plt.ylabel("Distance")
plt.show()

# --- Clustering hiérarchique avec liaison complète ---
Z_complete = linkage(data, method="complete")
plt.figure(figsize=(6, 4))
dendrogram(Z_complete, labels=labels)
plt.title("Clustering hiérarchique - Liaison complète (Complete Link)")
plt.xlabel("Points")
plt.ylabel("Distance")
plt.show()
