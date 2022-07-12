import matplotlib.pyplot as plt
import numpy as np


def quiver(dX, dY, path):
    fig, ax = plt.subplots(figsize=(64, 25))
    plt.xlim(0, 1024)
    plt.ylim(0, 370)

    grid_nums = [32, 65]
    X, Y = np.meshgrid(np.linspace(0, 1024, grid_nums[1], dtype="int"),
                       np.linspace(0, 496, grid_nums[0], dtype="int"))

    ax.quiver(X, 495-Y, dX, -dY)
    plt.savefig(path)


if __name__ == '__main__':
    dX_file = "../data/test/out/1_std_piv_dx.csv"
    dY_file = "../data/test/out/1_std_piv_dy.csv"
    dx = np.loadtxt(dX_file, delimiter=',')
    dy = np.loadtxt(dY_file, delimiter=',')

    quiver(dx, dy, "../data/test/std_piv/piv_1.png")

    print(0)
