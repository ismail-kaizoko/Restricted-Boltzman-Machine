from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt



def visualize_samples(images, num_samples=10):
    plt.figure(figsize=(20, 10))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()


def get_indices(item_list):
    matrix = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                   'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                   'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z'], dtype='<U1')
    # Use a list comprehension to find the index of each item in the matrix
    indices = [np.where(matrix == item)[0][0] for item in item_list]
    return indices
