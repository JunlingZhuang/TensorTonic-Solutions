import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here

    x_arr = np.asarray(x, dtype = float)

    z = 1.0 / (1.0 + np.exp(-x_arr))
    
    return z