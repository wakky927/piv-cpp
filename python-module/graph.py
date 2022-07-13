import matplotlib.pyplot as plt
import numpy as np


def quiver(da, path):
    # fig, ax = plt.subplots(figsize=(16, 16))
    # plt.xlim(0, 256)
    # plt.ylim(0, 256)
    # ax.quiver(da[:, 0], 255-da[:, 1], da[:, 2], -da[:, 3])

    fig, ax = plt.subplots(figsize=(64, 25))
    plt.xlim(-8, 760)
    plt.ylim(-8, 396)

    c = np.sqrt(da[:, 2]**2 + da[:, 3]**2)

    im = ax.quiver(da[:, 0], 396-da[:, 1], da[:, 2], -da[:, 3], c, cmap='jet')
    fig.colorbar(im)

    plt.savefig(path)


if __name__ == '__main__':
    file = "../data/test/out/C001H001S0001000001.csv"
    data = np.loadtxt(file, delimiter=',', skiprows=1)
    quiver(data, "../data/nn_test/piv_1.png")

    print(0)
