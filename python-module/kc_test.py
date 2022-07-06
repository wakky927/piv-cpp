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
                    [Modified KC Method]
                             |
                             V
                         [Results]

    """

    def __init__(self, img_path, out_path, filename, n):
        self.img_path = img_path
        self.out_path = out_path
        self.filename = filename
        self.n = n

    def run(self):
        for i in range(0, self.n):  # 1~n -> range(1, self.n + 1)
            # read img
            img_a = cv2.imread(self.img_path + self.filename + f"{i}.bmp", 0)  # 00000i -> {i:06}, 0000000i -> {i:08}
            img_b = cv2.imread(self.img_path + self.filename + f"{i + 1}.bmp", 0)

            # std piv (cross-correlation)
            G_dY, G_dX = self.std_piv(
                img_a=img_a,
                img_b=img_b,
                grid_nums=(65, 65),
                win_sizes=(16, 16),
                search_win_sizes=(10, 10),
                threshold=0.7
            )

            # pmc
            particle_position = self.pmc(img=img_a)

            # interpolation
            P_dYdX = self.interpolate_vectors(G_dY=G_dY, G_dX=G_dX, pp=particle_position)

            # KC method
            result = self.kc_method()

            return result

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
        :return: dY, dX: [2D ndarray (float), 2D ndarray (float)]: displacements
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

        return dY, dX

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

        tracer_img = np.zeros((15, 15), dtype=np.uint8)
        tracer_img = gauss_circle(image=tracer_img, sd=2)

        c = cv2.matchTemplate(img, tracer_img, cv2.TM_CCORR_NORMED)
        c_j, c_i = np.where(c > threshold)

        p_yx = sub_pixel_analysis(ji=[c_j, c_i], cc=c)

        return p_yx

    @staticmethod
    def interpolate_vectors(G_dY, G_dX, pp):
        """

        :param G_dY: [2D ndarray (float)] y-component of velocity field by PIV
        :param G_dX: [2D ndarray (float)] x-component of velocity field by PIV
        :param pp: [2D ndarray (float)] particle position
        :return: [2D ndarray (float)] estimated particle displacement

        [Coordinates]

            ----X   G_0(X, Y)          G_1(X+1, Y)
            |       --------------------
            Y       |                  |
                    |        * p(x, y) |
                    |                  |
                    |                  |
                    --------------------
                    G_3(X, Y+1)        G_2(X+1, Y+1)

        """
        def w(a, b, sigma):
            return np.exp(-(a**2 + b**2) / sigma**2)

        p_dydx = np.zeros_like(pp)
        p = 0

        for y, x in pp:
            G_0_Y = math.floor(y)
            G_0_X = math.floor(x)
            W = 0

            for j in range(1, 2):
                for i in range(1, 2):
                    p_dydx[p, 0] += G_dY[G_0_Y+j, G_0_X+i] * w(a=G_0_X+i-x, b=G_0_Y+j-y, sigma=0.5)
                    p_dydx[p, 1] += G_dX[G_0_Y+j, G_0_X+i] * w(a=G_0_X+i-x, b=G_0_Y+j-y, sigma=0.5)
                    W += w(a=G_0_X+i-x, b=G_0_Y+j-y, sigma=0.5)

            p_dydx[p] /= W
            p += 1

        return p_dydx

    def kc_method(self):
        pass


if __name__ == '__main__':
    pass
