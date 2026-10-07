import numpy as np
def conv2d(image: list, kernel: list, stride: int = 1, padding: int = 0) -> list:
    """
    Returns a two-dimensional list.
    """
    # Write code here
    img = np.asarray(image, dtype = np.float64)
    kernel = np.asarray(kernel, dtype = np.float64)
    i_H, i_W = img.shape
    k_H, k_W = kernel.shape
    o_H, o_W = getOutPutDim(i_H, i_W, k_H, k_W, stride, padding)
    img = np.pad(img, pad_width = ((padding, padding), (padding, padding)))
    output = np.zeros((o_H, o_W))
    for i in range(o_H):
        for j in range(o_W):
            img_copy = img[i * stride : i * stride + k_H, j * stride : j * stride + k_W]
            output[i, j] = (img_copy * kernel).sum()

    return output.tolist()
            
    


def getOutPutDim(i_H, i_W, k_H, k_W, stride, padding):
    o_H = math.floor((i_H - k_H + 2 * padding) / stride) + 1 
    o_W = math.floor((i_W - k_W + 2 * padding) / stride) + 1
    return (o_H, o_W)