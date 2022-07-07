import matplotlib.pyplot as plt
import numpy as np


def quiver(da, path):
    fig, ax = plt.subplots(figsize=(16, 16))

    ax.quiver(da[:, 1], 255-da[:, 0], da[:, 3], -da[:, 2])
    plt.savefig(path)


if __name__ == '__main__':
    file = "../data/kc-test/piv_1.csv"
    data = np.loadtxt(file, delimiter=',', skiprows=1)
    quiver(data, "../data/kc-test/piv_1.png")

    print(0)
