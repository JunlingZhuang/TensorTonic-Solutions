import numpy as np

# update sample to corresponding clusters  
def update(X, labels, C):
    new_centroids = C.copy() #k, d
    for j in range(C.shape[0]):
        mask = (labels == j) # n,
        if mask.any():
            new_centroids[j] = X[mask].mean(axis = 0)
    return new_centroids # k, d

# caculate the distance from this sample to 
def pairwise_sq_dist(X, C):
    # X (n, d) ->(n, 1, d) C(k, d) -> (1, k, d)
    # (n, k, d)
    diff = X[:, None, :] - C[None, :, :]
    dist = (diff ** 2).sum(axis = -1)
    return dist

def assign(X, C):
    dist = pairwise_sq_dist(X, C) # n, k 
    labels = dist.argmin(axis = -1)
    return labels # (n, )

def kmeans(X: list, k: int, max_iters: int = 100, seed: int = 42) -> tuple:
    """
    Returns cluster labels and the final centroid matrix.
    """
    X = np.asarray(X, dtype = np.float64)
    n, d = X.shape
    rng = np.random.RandomState(seed)
    centroids = X[rng.choice(n, size = k, replace= False)].copy() # k, d

    for _ in range(max_iters):
        labels = assign(X, centroids) # n,
        new_centroids = update(X, labels, centroids)
        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    labels = assign(X, centroids)
    inertia = pairwise_sq_dist(X, centroids)[np.arange(n), labels].sum()
    return labels.tolist(), np.round(centroids, 4).tolist()