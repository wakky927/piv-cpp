import matplotlib.pyplot as plt
import numpy as np


def quiver(dX, dY, dX_2, dY_2, path):
    fig, ax = plt.subplots(figsize=(64, 25))
    plt.xlim(-8, 1032)
    plt.ylim(-8, 504)

    grid_nums = [32, 65]  # 16, 33
    X, Y = np.meshgrid(np.linspace(0, 1024, grid_nums[1], dtype="int"),
                       np.linspace(0, 496, grid_nums[0], dtype="int"))

    # ax.quiver(X, 495-Y, dX, -dY, color='r')
    ax.quiver(X, 495-Y, dX_2, -dY_2, color='k')
    plt.savefig(path)


if __name__ == '__main__':
    dX_file = "../data/test/out/1_std_piv_dx.csv"
    dY_file = "../data/test/out/1_std_piv_dy.csv"
    dX_2_file = "../data/test/out/1_std_piv_dx_2.csv"
    dY_2_file = "../data/test/out/1_std_piv_dy_2.csv"

    dx = np.loadtxt(dX_file, delimiter=',')
    dy = np.loadtxt(dY_file, delimiter=',')
    dx_2 = np.loadtxt(dX_2_file, delimiter=',')
    dy_2 = np.loadtxt(dY_2_file, delimiter=',')

    quiver(dx, dy, dx_2, dy_2, "../data/test/std_piv/piv_1.png")

    print(0)
