import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here

    N, D = X.shape
    w = np.zeros(D)
    b = 0

    for _ in range(steps):
        z = X @ w + b
        p = _sigmoid(z)
        error  = p - y
        dw = X.T @ error / N
        db = error.sum() / N

        w = w - lr * dw
        b = b - lr * db

    return w, b