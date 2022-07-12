import matplotlib.pyplot as plt
import numpy as np


def quiver(da, path):
    fig, ax = plt.subplots(figsize=(16, 16))
    plt.xlim(0, 256)
    plt.ylim(0, 256)

    ax.quiver(da[:, 0], 255-da[:, 1], da[:, 2], -da[:, 3])
    plt.savefig(path)


if __name__ == '__main__':
    file = "../data/nn_test/piv_1.csv"
    data = np.loadtxt(file, delimiter=',', skiprows=1)
    quiver(data, "../data/nn_test/piv_1.png")

    print(0)
