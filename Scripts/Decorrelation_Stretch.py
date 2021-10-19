import numpy as np
import cv2
from functools import reduce
def decorrstretch(img, tol=None):
    original_shape = img.shape
    img = img.reshape((-1, 3)).astype(np.float)

    cov = np.cov(img.T)
    sigma = np.diag(np.sqrt(cov.diagonal()))

    eigval, eigvec = np.linalg.eig(cov)
    strech_matrix = np.diag(1/np.sqrt(eigval))

    mean = np.mean(img, axis=0)
    img -= mean

    T = reduce(np.dot, [sigma, eigvec, strech_matrix, eigvec.T])
    offset = mean - np.dot(mean, T)
    img = np.dot(img, T)
    img += mean
    img += offset
    output = img.reshape(original_shape)
    print(output)
    for i in range(3):
        output[:, :, i] = 255 * (output[:, :, i] - output[:, :, i].min()) / (output[:, :, i].max() - output[:, :, i].min())
    print("as", output)
    return output.astype(np.uint8)