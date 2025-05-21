# Save as hmrf_em.py

import numpy as np
from scipy.ndimage import convolve
from sklearn.cluster import KMeans

def normalize(image):
    image = image.astype(np.float32)
    image -= np.min(image)
    image /= np.max(image) + 1e-6
    return image

def kmeans_init(image, mask, n_classes):
    X = image[mask > 0].reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_classes, n_init=5).fit(X)
    labels = np.zeros_like(image, dtype=np.int32)
    labels[mask > 0] = kmeans.labels_ + 1  # Avoid 0 as label
    return labels

def run_hmrf_em(image, mask, init_labels, n_classes=6, beta=1.0, max_iter=10):
    """
    image: 3D np.array (normalized intensity)
    mask: 3D binary np.array (brain mask)
    init_labels: 3D np.array of initial labels
    """

    mu = np.zeros(n_classes)
    sigma2 = np.ones(n_classes)

    # Estimate initial means and variances
    for k in range(n_classes):
        region = (init_labels == (k + 1)) & (mask > 0)
        if np.any(region):
            mu[k] = image[region].mean()
            sigma2[k] = image[region].var()

    labels = init_labels.copy()
    struct = np.ones((3, 3, 3))

    for it in range(max_iter):
        print(f"[HMRF] Iteration {it+1}/{max_iter}")
        unary = np.zeros((n_classes,) + image.shape)

        for k in range(n_classes):
            diff2 = (image - mu[k])**2
            unary[k] = 0.5 * np.log(sigma2[k]) + diff2 / (2 * sigma2[k])

        spatial = np.zeros_like(unary)

        for k in range(n_classes):
            neighbor = convolve((labels == (k + 1)).astype(np.float32), struct, mode='constant')
            spatial[k] = -beta * neighbor

        energy = unary + spatial

        new_labels = np.argmin(energy, axis=0) + 1  # shift back to 1-based labels
        labels = np.where(mask > 0, new_labels, 0)

        # Update mu and sigma2
        for k in range(n_classes):
            region = (labels == (k + 1)) & (mask > 0)
            if np.any(region):
                mu[k] = image[region].mean()
                sigma2[k] = image[region].var()

    return labels
