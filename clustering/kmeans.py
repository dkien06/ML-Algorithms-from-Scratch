
import numpy as np
import matplotlib.pyplot as plt


def initialize_centroids(X, k):
    """Randomly select k samples as initial centroids."""
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices].copy()


def calculate_distance(x1, x2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(x1 - x2)


def assign_labels(X, centroids):
    """Assign each sample to its nearest centroid."""
    distances = np.linalg.norm(
        X[:, np.newaxis, :] - centroids[np.newaxis, :, :],
        axis=2
    )
    return np.argmin(distances, axis=1)


def calculate_new_centroids(X, labels, centroids, k):
    """Calculate new centroids, preserving empty clusters."""
    new_centroids = centroids.copy()

    for i in range(k):
        cluster = X[labels == i]

        if len(cluster) > 0:
            new_centroids[i] = np.mean(cluster, axis=0)

    return new_centroids


def k_means(X, k, max_iters=1000, tol=1e-4):
    """
    K-Means clustering.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    k : int
        Number of clusters.
    max_iters : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    centroids : ndarray, shape (k, n_features)
    labels : ndarray, shape (n_samples,)
    """
    if k <= 0 or k > X.shape[0]:
        raise ValueError("k must be between 1 and the number of samples.")

    centroids = initialize_centroids(X, k)

    for iteration in range(max_iters):
        labels = assign_labels(X, centroids)

        new_centroids = calculate_new_centroids(
            X, labels, centroids, k
        )

        if np.allclose(centroids, new_centroids, atol=tol):
            centroids = new_centroids
            break

        centroids = new_centroids

    labels = assign_labels(X, centroids)

    return centroids, labels


def create_dummy_data():
    cluster1 = np.random.randn(100, 2) + np.array([2, 2])
    cluster2 = np.random.randn(100, 2) + np.array([8, 3])
    cluster3 = np.random.randn(100, 2) + np.array([5, 8])

    return np.vstack([cluster1, cluster2, cluster3])


def plot_clusters(X, labels, centroids):
    plt.figure(figsize=(8, 6))

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30)
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        color="red",
        s=200,
        label="Centroids"
    )

    plt.title("K-Means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


def main():
    X = create_dummy_data()

    centroids, labels = k_means(X, k=3)

    print("Centroids:")
    print(centroids)

    plot_clusters(X, labels, centroids)


if __name__ == "__main__":
    main()
