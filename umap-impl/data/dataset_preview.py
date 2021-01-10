import numpy as np
import matplotlib.pyplot as plt
import sys

from data.datasets import Mnist, FashionMnist, Cifar10, Cifar100, Fmow
from params import mnist_category_names, fashion_mnist_category_names, cifar10_category_names, cifar100_category_names, \
    fmow_category_names


# mnist_preview function create subplot of chosen datapoints in mnist dataset

def mnist_preview():
    dataset = Mnist()
    dataset.load()

    data, labels = dataset.get()
    # print(len(data)) 70000

    data_index = 0

    fig_size = (4, 4)
    fig_shape = (2, 2)

    dpi = 300
    axis_size = 5

    plt.rc("xtick", labelsize=axis_size)
    plt.rc("ytick", labelsize=axis_size)

    fig = plt.figure(figsize=fig_size, dpi=dpi)

    n_rows = fig_shape[0]
    n_col = fig_shape[1]
    x_pos = 0
    y_pos = 0
    for i in range(1, n_col * n_rows + 1):
        if x_pos >= n_col:
            x_pos = 0
            y_pos += 1

        img_size = int(np.sqrt(len(data[data_index])))

        img = data[data_index].reshape(img_size, img_size)

        plt.subplot(n_rows, n_col, i)
        plt.imshow(img, cmap="binary")
        plt.title(mnist_category_names[labels[data_index]], fontsize=8)

        x_pos += 1
        data_index += 17500

    fig.tight_layout()
    plt.show()

# fashion_mnist_preview function create subplot of chosen datapoints in fashion_mnist dataset

def fashion_mnist_preview():
    dataset = FashionMnist()
    dataset.load()

    data, labels = dataset.get()
    # print(len(data)) 70000

    data_index = 0

    fig_size = (4, 4)
    fig_shape = (2, 2)

    dpi = 300
    axis_size = 5

    plt.rc("xtick", labelsize=axis_size)
    plt.rc("ytick", labelsize=axis_size)

    fig = plt.figure(figsize=fig_size, dpi=dpi)

    n_rows = fig_shape[0]
    n_col = fig_shape[1]
    x_pos = 0
    y_pos = 0
    for i in range(1, n_col * n_rows + 1):
        if x_pos >= n_col:
            x_pos = 0
            y_pos += 1

        img_size = int(np.sqrt(len(data[data_index])))

        img = data[data_index].reshape(img_size, img_size)

        plt.subplot(n_rows, n_col, i)
        plt.imshow(img, cmap="gist_gray")
        plt.title(fashion_mnist_category_names[labels[data_index]], fontsize=8)

        x_pos += 1
        data_index += 10000

    fig.tight_layout()
    plt.show()

# cifar10_preview function create subplot of chosen datapoints in cifar10 dataset

def cifar10_preview():
    dataset = Cifar10()
    dataset.load()

    data, labels = dataset.get()
    # print(len(data)) 60000

    data_index = 0

    fig_size = (4, 4)
    fig_shape = (2, 2)

    dpi = 300
    axis_size = 5

    plt.rc("xtick", labelsize=axis_size)
    plt.rc("ytick", labelsize=axis_size)

    fig = plt.figure(figsize=fig_size, dpi=dpi)

    n_rows = fig_shape[0]
    n_col = fig_shape[1]
    x_pos = 0
    y_pos = 0
    for i in range(1, n_col * n_rows + 1):
        if x_pos >= n_col:
            x_pos = 0
            y_pos += 1

        img_size = int(np.sqrt(len(data[data_index]) / 3))
        arr_len = int(len(data[data_index]) / 3)

        r = data[data_index][0:arr_len].reshape(img_size, img_size)
        g = data[data_index][arr_len:(arr_len * 2)].reshape(img_size, img_size)
        b = data[data_index][(arr_len * 2):].reshape(img_size, img_size)
        img = np.dstack((r, g, b))

        plt.subplot(n_rows, n_col, i)
        plt.imshow(img)
        plt.title(cifar10_category_names[labels[data_index]], fontsize=8)

        x_pos += 1
        data_index += 10000

    fig.tight_layout()
    plt.show()

# cifar100_preview function create subplot of chosen datapoints in cifar100 dataset

def cifar100_preview():
    dataset = Cifar100()
    dataset.load()

    data, labels = dataset.get()
    # print(len(data)) 60000

    data_index = 0

    fig_size = (4, 4)
    fig_shape = (2, 2)

    dpi = 300
    axis_size = 5

    plt.rc("xtick", labelsize=axis_size)
    plt.rc("ytick", labelsize=axis_size)

    fig = plt.figure(figsize=fig_size, dpi=dpi)

    n_rows = fig_shape[0]
    n_col = fig_shape[1]
    x_pos = 0
    y_pos = 0
    for i in range(1, n_col * n_rows + 1):
        if x_pos >= n_col:
            x_pos = 0
            y_pos += 1

        img_size = int(np.sqrt(len(data[data_index]) / 3))
        arr_len = int(len(data[data_index]) / 3)

        r = data[data_index][0:arr_len].reshape(img_size, img_size)
        g = data[data_index][arr_len:(arr_len * 2)].reshape(img_size, img_size)
        b = data[data_index][(arr_len * 2):].reshape(img_size, img_size)
        img = np.dstack((r, g, b))

        plt.subplot(n_rows, n_col, i)
        plt.imshow(img)
        plt.title(cifar100_category_names[labels[data_index]], fontsize=8)

        x_pos += 1
        data_index += 10000

    fig.tight_layout()
    plt.show()

# fmow_preview function create subplot of chosen datapoints in fmow dataset

def fmow_preview():
    dataset = Fmow("val_rgb")

    data, labels = dataset.get()

    # print(len(data)) 363572 for train_rgb and 53041 for val_rgb
    data_index = 0

    fig_size = (4, 4)
    fig_shape = (2, 2)

    dpi = 300
    axis_size = 5

    plt.rc("xtick", labelsize=axis_size)
    plt.rc("ytick", labelsize=axis_size)

    fig = plt.figure(figsize=fig_size, dpi=dpi)

    n_rows = fig_shape[0]
    n_col = fig_shape[1]
    x_pos = 0
    y_pos = 0
    for i in range(1, n_col * n_rows + 1):
        if x_pos >= n_col:
            x_pos = 0
            y_pos += 1

        img_size = int(np.sqrt(len(data[data_index]) / 3))
        arr_len = int(len(data[data_index]) / 3)
        rgb_arr = np.zeros(shape=(img_size, img_size, 3), dtype="uint8")

        count = 0
        for pixel_height in range(0, img_size):
            for pixel_width in range(0, img_size):
                rgb_arr[pixel_height][pixel_width][0] = data[data_index][count]
                rgb_arr[pixel_height][pixel_width][1] = data[data_index][arr_len + count]
                rgb_arr[pixel_height][pixel_width][2] = data[data_index][arr_len * 2 + count]
                count += 1

        plt.subplot(n_rows, n_col, i)
        plt.imshow(rgb_arr)
        plt.title(fmow_category_names[labels[data_index]], fontsize=8)

        x_pos += 1
        data_index += 10000

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Test dataset preview:")
    fmow_preview()
    # sys.exit()
