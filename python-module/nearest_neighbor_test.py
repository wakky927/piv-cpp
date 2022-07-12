from decimal import Decimal, ROUND_HALF_UP
import math

import cv2
import numpy as np
from scipy.stats import norm


class KC(object):
    """
    [URL] https://doi.org/10.1007/s003480070005

    [Flow Chart]

                     [Particle image]
                            |
             ---------------------------------
             |                               |
             V                               V
        [A standard                  [Particle Mask
         Correlation PIV]             Correlation Method]
             |                               |
             V                               V
        [Regular Grid                [Individual Particle
         Velocity Field]              Image Information  ]
             |                               |
             ---------------------------------
                             |
                             V
             [Interpolation of Initial Velocity
              Vectors for Each Particle        ]
                             |
                             V
                    [Nearest-neighbor Method]
                             |
                             V
                         [Results]

    """

    def __init__(self, img_path, out_path, filename, fmt, n):
        self.img_path = img_path
        self.out_path = out_path
        self.filename = filename
        self.fmt = fmt
        self.n = n

    def run(self):
        for i in range(1, self.n+1):  # 0 ~ n-1 -> range(0, self.n)
            # read img
            img_a = cv2.imread(f"{self.img_path}{self.filename}{i}.{self.fmt}", 0)  # 00000i -> {i:06}
            img_b = cv2.imread(f"{self.img_path}{self.filename}{i+1}.{self.fmt}", 0)

            # std piv (cross-correlation)
            G_Y, G_X, G_dY, G_dX = self.std_piv(
                img_a=img_a,
                img_b=img_b,
                grid_nums=(65, 65),
                win_sizes=(16, 16),
                search_win_sizes=(10, 10),
                threshold=0.5
            )

            # pmc
            particle_position_a = self.pmc(img=img_a, threshold=0.5)
            particle_position_b = self.pmc(img=img_b, threshold=0.5)

            # interpolation
            P_dYdX = self.interpolate_vectors(G_Y=G_Y, G_X=G_X, G_dY=G_dY, G_dX=G_dX, pp=particle_position)

            # KC method
            result = self.kc_method(pp_a=particle_position_a, pp_b=particle_position_b, P_dYdX=P_dYdX)

            # save result
            np.savetxt(f"{self.out_path}{self.filename}{i}.csv", result, delimiter=',', header="y, x, dY, dX")

    @staticmethod
    def std_piv(img_a, img_b, grid_nums, win_sizes, search_win_sizes, threshold=0.7):
        """
        Standard PIV (Cross-correlation)
        :param img_a: [2D ndarray (uint8)] 1st img
        :param img_b: [2D ndarray (uint8)] 2nd img
        :param grid_nums: [tuple(int, int)] grid numbers: (N_Y, N_X)
        :param win_sizes: [tuple(int, int)] window sizes: (H, W) -> 2H * 2W
        :param search_win_sizes: [tuple(int, int)] search window sizes: (S_Y, S_X) -> H + S_Y, W + S_X
        :param threshold: [float] threshold (default 0.7)
        :return: Y, X, dY, dX: [2D ndarray, 2D ndarray, 2D ndarray, 2D ndarray]: displacements
        """
        height, width = img_a.shape

        img1 = np.zeros((height + 2 * (search_win_sizes[0] + win_sizes[0]),
                         width + 2 * (search_win_sizes[1] + win_sizes[1]))).astype(np.uint8)
        img2 = np.zeros((height + 2 * (search_win_sizes[0] + win_sizes[0]),
                         width + 2 * (search_win_sizes[1] + win_sizes[1]))).astype(np.uint8)

        img1[search_win_sizes[0] + win_sizes[0]:search_win_sizes[0] + win_sizes[0] + height,
             search_win_sizes[1] + win_sizes[1]:search_win_sizes[1] + win_sizes[1] + width] = img_a
        img2[search_win_sizes[0] + win_sizes[0]:search_win_sizes[0] + win_sizes[0] + height,
             search_win_sizes[1] + win_sizes[1]:search_win_sizes[1] + win_sizes[1] + width] = img_b

        X, Y = np.meshgrid(np.linspace(0, width, grid_nums[1], dtype="int"),
                           np.linspace(0, height, grid_nums[0], dtype="int"))
        dX = np.zeros(grid_nums)
        dY = np.zeros(grid_nums)

        for y in range(0, grid_nums[0], 1):
            for x in range(0, grid_nums[1], 1):
                i = X[y, x] + search_win_sizes[1] + win_sizes[1]
                j = Y[y, x] + search_win_sizes[0] + win_sizes[0]

                iw1 = img1[j - win_sizes[0]:j + win_sizes[0], i - win_sizes[1]:i + win_sizes[1]]
                iw2 = img2[j - win_sizes[0] - search_win_sizes[0]:j + win_sizes[0] + search_win_sizes[0],
                           i - win_sizes[1] - search_win_sizes[1]:i + win_sizes[1] + search_win_sizes[1]]

                cc = cv2.matchTemplate(iw2, iw1, cv2.TM_CCOEFF_NORMED)
                cc_max = np.max(cc)

                if cc_max > threshold:
                    y_max, x_max = np.unravel_index(np.argmax(cc),
                                                    (2 * search_win_sizes[0] + 1, 2 * search_win_sizes[1] + 1))

                    # sub-pixel interpolation
                    if x_max == 0 or x_max == 2 * search_win_sizes[1] or y_max == 0 or y_max == 2 * search_win_sizes[0]:
                        x_sub = 0
                        y_sub = 0
                    else:
                        cc_center = cc[y_max, x_max]
                        cc_top = cc[y_max - 1, x_max]
                        cc_bottom = cc[y_max + 1, x_max]
                        cc_left = cc[y_max, x_max - 1]
                        cc_right = cc[y_max, x_max + 1]

                        if np.any(np.array([cc_center, cc_top, cc_bottom, cc_left, cc_right]) <= 0):
                            x_sub = 0
                            y_sub = 0
                        else:
                            x_sub = (np.log(cc_left) - np.log(cc_right)) \
                                    / (2 * (np.log(cc_left) + np.log(cc_right) - 2 * np.log(cc_center)))
                            y_sub = (np.log(cc_top) - np.log(cc_bottom)) \
                                    / (2 * (np.log(cc_top) + np.log(cc_bottom) - 2 * np.log(cc_center)))

                    dX[y, x] = x_max + x_sub - search_win_sizes[1]
                    dY[y, x] = y_max + y_sub - search_win_sizes[0]
                else:
                    dX[y, x] = np.nan
                    dY[y, x] = np.nan

        return Y, X, dY, dX

    @staticmethod
    def pmc(img, threshold=0.7):
        def gauss_circle(image, sd, high=255, low=0, random_sd=0):
            height, width = image.shape[:2]
            rd = np.random.normal(0, random_sd, (height, width))
            scale = high - low
            s = 1 / norm.pdf(0, loc=0, scale=sd)

            for y in range(0, height):
                for x in range(0, width):
                    dist = np.sqrt((x - width // 2) ** 2 + (y - height // 2) ** 2)
                    n = norm.pdf(dist, loc=0, scale=sd)
                    image[y, x] = int(np.clip(n * s * scale + low + rd[y, x], 0, 255))
            return image

        def sub_pixel_analysis(ji, cc):
            res = np.zeros((1, 2))

            for j, i in zip(ji[0], ji[1]):
                if 0 < j < cc.shape[0] - 1 and 0 < i < cc.shape[1] - 1:
                    # judge whether peak value or not among 3x3
                    if cc[j, i] == np.amax(cc[j-1:j+2, i-1:i+2]):
                        eps_x = 0.5 * (np.log(cc[j, i - 1]) - np.log(cc[j, i + 1])) \
                                / (np.log(cc[j, i - 1]) + np.log(cc[j, i + 1]) - 2 * np.log(cc[j, i]))
                        eps_y = 0.5 * (np.log(cc[j - 1, i]) - np.log(cc[j + 1, i])) \
                                / (np.log(cc[j - 1, i]) + np.log(cc[j + 1, i]) - 2 * np.log(cc[j, i]))

                        # round half-up
                        eps_x = float(Decimal(str(eps_x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))
                        eps_y = float(Decimal(str(eps_y)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))

                        res = np.vstack([res, [j + eps_y, i + eps_x]])

            return np.unique(res[1:, :], axis=0)  # Exclude duplicates just in case...

        tracer_img = np.zeros((9, 9), dtype=np.uint8)
        tracer_img = gauss_circle(image=tracer_img, sd=2)

        c = cv2.matchTemplate(img, tracer_img, cv2.TM_CCORR_NORMED)
        c_j, c_i = np.where(c > threshold)

        p_yx = sub_pixel_analysis(ji=[c_j, c_i], cc=c)

        return p_yx

    @staticmethod
    def interpolate_vectors(G_Y, G_X, G_dY, G_dX, pp):
        """
        Interpolation vectors of particle displacement
        :param G_Y: [2D ndarray] y-coordinates
        :param G_X: [2D ndarray] x-coordinates
        :param G_dY: [2D ndarray] y-component of velocity field by PIV
        :param G_dX: [2D ndarray] x-component of velocity field by PIV
        :param pp: [2D ndarray] particle position
        :return: [2D ndarray] estimated particle displacement

        [Coordinates]

            ----I
            |       (I, J) = (X_G0, Y_G0)  (I+1, J) = (X_G1, Y_G1)
            J             O--------------------O
                          |                    |
                          |        * p(x, y)   |
                          |                    |
                          |                    |
                          |                    |
                          O--------------------O
                    (I, J+1) = (X_G3, Y_G3)  (I+1, J+1) = (X_G2, Y_G2)

        """
        def w(a, b, sigma):
            return np.exp(-(a**2 + b**2) / sigma**2)

        p_dydx = np.zeros_like(pp)
        p = 0

        for y, x in pp:
            # init weight
            W = 0

            # get x-y coordinates
            X, Y = G_X[0, :], G_Y[:, 0]

            # get (I, J)
            I = np.where(X == X[(X <= x)].max())[0]
            J = np.where(Y == Y[(Y <= y)].max())[0]

            # interpolation
            for j in range(0, 2):
                for i in range(0, 2):
                    if np.isnan(G_dX[J+j, I+i]) or np.isnan(G_dY[J+j, I+i]):
                        pass
                    else:
                        p_dydx[p, 0] += G_dY[J+j, I+i] * w(a=X[I+i]-x, b=Y[J+j]-y, sigma=2)
                        p_dydx[p, 1] += G_dX[J+j, I+i] * w(a=X[I+i]-x, b=Y[J+j]-y, sigma=2)
                        W += w(a=X[I+i]-x, b=Y[J+j]-y, sigma=2)

            if W == 0:
                p_dydx[p] = np.nan
            else:
                p_dydx[p] /= W

            p += 1

        return p_dydx

    @staticmethod
    def nn_method(pp_a, pp_b, P_dYdX):
        r = np.hstack([pp, P_dYdX])

        return r


if __name__ == '__main__':
    kc = KC(img_path="../data/std-piv-imgs/", out_path="../data/kc-test/", filename="piv_", fmt="bmp", n=2)
    kc.run()

    print(0)
