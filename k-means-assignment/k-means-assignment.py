import numpy as np
def k_means_assignment(points: list, centroids: list) -> list:
    """
    Returns the nearest-centroid index for every point.
    """
    # Write code here
    points = np.asarray(points, dtype = np.float64)
    centroids = np.asarray(centroids, dtype = np.float64)
    diff = points[:, None, :] - centroids[None, :, :]
    dist = (diff ** 2).sum(axis = - 1) # n, k
    labels = dist.argmin(axis = -1)
    return labels.tolist()