"""
Author: Tobias Morocutti
Matr.Nr.: K12008172
Exercise 5
"""

import dill as pkl
import h5py
import matplotlib.pyplot as plt


def write_dic_to_pickle_file(data, filepath, mode='wb'):
    with open(filepath, mode) as fh:
        pkl.dump(data, file=fh)


def read_pickle_file(path):
    with open(path, 'rb') as pfh:
        data = pkl.load(pfh)

    return data


def read_h5py_file(path):
    with h5py.File(path, 'r') as f:
        return f


def visualize_images(input_image, known_image, target_image, pred_image):
    fig = plt.figure(figsize=(10, 7))

    # input image
    fig.add_subplot(2, 2, 1)
    plt.imshow(input_image, cmap='gray')
    plt.axis('off')
    plt.title("Input Image")

    # known image
    fig.add_subplot(2, 2, 2)
    plt.imshow(known_image, cmap='gray')
    plt.axis('off')
    plt.title("Known Image")

    # target image
    fig.add_subplot(2, 2, 3)
    plt.imshow(target_image, cmap='gray')
    plt.axis('off')
    plt.title("Target Image")

    # prediction image
    fig.add_subplot(2, 2, 4)
    plt.imshow(pred_image, cmap='gray')
    plt.axis('off')
    plt.title("My Prediciton Image")

    plt.show(block=True)

