from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP

import cv2
import numpy as np
from scipy.ndimage.filters import convolve, maximum_filter
from scipy.stats import norm


class KC(object):
    """
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
        [Regular Grid                [Post processing]
         Velocity Field]                     |
             |                               |
             ---------------------------------
                             |
                             V
             [Interpolation of Initial Velocity
              Vectors for Each Particle        ]
                             |
                             V
                    [nearest-neighbor Method]
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
        for i in range(1, self.n):  # 0 ~ n-1 -> range(0, self.n)
            # read img
            img_a = cv2.imread(f"{self.img_path}{self.filename}{i:06}.{self.fmt}", 0)  # 00000i -> {i:06}
            img_b = cv2.imread(f"{self.img_path}{self.filename}{i + 1:06}.{self.fmt}", 0)

            # pre-processing

            # gaussian blur
            img_a_gauss = cv2.GaussianBlur(img_a, (3, 3), 3)
            img_b_gauss = cv2.GaussianBlur(img_b, (3, 3), 3)

            roberts_cross_v = np.array([[1, 0],
                                        [0, -1]])

            roberts_cross_h = np.array([[0, 1],
                                        [-1, 0]])

            img_a_roberts = img_a.astype(np.float64) / 255.0
            img_b_roberts = img_b.astype(np.float64) / 255.0

            vertical = convolve(img_a_roberts, roberts_cross_v)
            horizontal = convolve(img_a_roberts, roberts_cross_h)
            edged_img_a = np.sqrt(np.square(horizontal) + np.square(vertical)) * 255

            vertical = convolve(img_b_roberts, roberts_cross_v)
            horizontal = convolve(img_b_roberts, roberts_cross_h)
            edged_img_b = np.sqrt(np.square(horizontal) + np.square(vertical)) * 255

            cv2.imwrite(f"{self.out_path}{self.filename}{i}_e.png", edged_img_a)

            # high-pass filter
            kernel_high_pass = np.array([[-1, -1, -1],
                                         [-1,  8, -1],
                                         [-1, -1, -1]], np.float32)
            img_a_high_pass = cv2.filter2D(img_a_gauss, -1, kernel_high_pass)
            img_b_high_pass = cv2.filter2D(img_b_gauss, -1, kernel_high_pass)

            # img_a_high_pass = self.highpass_filter(src=img_a_gauss)
            # img_b_high_pass = self.highpass_filter(src=img_b_gauss)

            cv2.imwrite(f"{self.out_path}{self.filename}{i}_highpass.png", img_a_high_pass)

            # 1st std piv (cross-correlation)
            G_X, G_Y, G_dX, G_dY, flag = self.piv(
                img_a=edged_img_a,
                img_b=edged_img_b,
                grid_nums=(32, 65),  # 32, 65 (16, 33)
                win_sizes=(16, 16),  # 16, 16 (32, 32)
                search_win_sizes=(8, 8),  # 8, 8 (16, 16)
                threshold=0.5,
            )

            np.savetxt(f"{self.out_path}{self.filename}{i}_std_piv_x.csv", G_X, delimiter=',')
            np.savetxt(f"{self.out_path}{self.filename}{i}_std_piv_y.csv", G_Y, delimiter=',')
            np.savetxt(f"{self.out_path}{self.filename}{i}_std_piv_dx.csv", G_dX, delimiter=',')
            np.savetxt(f"{self.out_path}{self.filename}{i}_std_piv_dy.csv", G_dY, delimiter=',')

            # post-processing
            G_dX, G_dY = self.post(dX=G_dX, dY=G_dY, flag=flag)

            np.savetxt(f"{self.out_path}{self.filename}{i}_std_piv_dx_2.csv", G_dX, delimiter=',')
            np.savetxt(f"{self.out_path}{self.filename}{i}_std_piv_dy_2.csv", G_dY, delimiter=',')

            # pmc
            particle_position_a = self.pmc(img=img_a, threshold=0.5)
            particle_position_b = self.pmc(img=img_b, threshold=0.5)

            img_a_2 = np.zeros_like(img_a)
            img_b_2 = np.zeros_like(img_b)

            for s, t in zip(particle_position_a[:, 0].astype(int), particle_position_a[:, 1].astype(int)):
                img_a_2[t, s] = 255
            for s, t in zip(particle_position_a[:, 0].astype(int), particle_position_a[:, 1].astype(int)):
                img_b_2[t, s] = 255

            cv2.imwrite(f"{self.out_path}{self.filename}{i}.bmp", img_a_2)
            cv2.imwrite(f"{self.out_path}{self.filename}{i + 1}.bmp", img_b_2)

            # interpolation
            P_dXdY = self.interpolate_vectors(G_X=G_X, G_Y=G_Y, G_dX=G_dX, G_dY=G_dY, pp=particle_position_a)

            img_a_b = np.zeros_like(img_a)
            pp = particle_position_a + P_dXdY
            for s, t in zip(pp[:, 0].astype(int), pp[:, 1].astype(int)):
                if 0 <= s < 256 and 0 <= t < 256:
                    img_a_b[t, s] = 255
            cv2.imwrite(f"{self.out_path}{self.filename}{i + 1}_pred.bmp", img_a_b)

            # nearest-neighbor method
            result = self.nn_method(pp_a=particle_position_a, pp_b=particle_position_b, P_dXdY=P_dXdY)

            # save result
            np.savetxt(f"{self.out_path}{self.filename}{i}.csv", result, delimiter=',', header="x, y, dX, dY")

    @staticmethod
    def highpass_filter(src, a=0.5):
        # 2D FFT
        src = np.fft.fft2(src)

        h, w = src.shape
        cy, cx = int(h/2), int(w/2)  # center
        rh, rw = int(a*cy), int(a*cx)  # filter size

        f_src = np.fft.fftshift(src)  # shift quadrant (first <-> third, second <-> forth)

        f_dst = f_src.copy()
        f_dst[cy-rh:cy+rh, cx-rw:cx+rw] = 0  # highpass filtering

        f_dst = np.fft.fftshift(f_dst)  # reset quadrant

        # inverse FFT
        dst = np.fft.ifft2(f_dst)
        dst = dst.real

        return dst.astype(np.uint8)

    @staticmethod
    def detect_second_peak(arr_2d, filter_size, th):
        peaks = deepcopy(arr_2d)

        local_max = maximum_filter(arr_2d, footprint=np.ones((filter_size, filter_size)), mode='constant')
        np.place(peaks, arr_2d != local_max, 0)
        np.place(peaks, peaks < th * 0.8, 0)

        if np.partition(peaks.flatten(), -2)[-2] <= 0:
            return np.nan, np.nan

        second_peak_index = np.unravel_index(np.argpartition(peaks.flatten(), -2)[-2], peaks.shape)

        return second_peak_index

    @staticmethod
    def piv(img_a, img_b, grid_nums, win_sizes, search_win_sizes, threshold=0.7, r_mode=False, dx=None, dy=None):
        """
        Standard PIV (Cross-correlation)
        :param img_a: [2D ndarray (uint8)] 1st img
        :param img_b: [2D ndarray (uint8)] 2nd img
        :param grid_nums: [tuple(int, int)] grid numbers: (N_Y, N_X)
        :param win_sizes: [tuple(int, int)] window sizes: (H, W) -> 2H * 2W
        :param search_win_sizes: [tuple(int, int)] search window sizes: (S_Y, S_X) -> H + S_Y, W + S_X
        :param threshold: [float] threshold (default 0.7)
        :param r_mode: [bool] recursive mode (default False)
        :param dx: [2D ndarray (float)] displacements
        :param dy: [2D ndarray (float)] displacements

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
        flag = np.zeros(grid_nums, dtype=int)

        if not r_mode:
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
                        y_max, x_max = np.unravel_index(np.argmax(cc), cc.shape)

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
                        flag[y, x] = 1

        elif r_mode and dx is not None and dy is not None:
            for y in range(0, grid_nums[0], 1):
                for x in range(0, grid_nums[1], 1):
                    i = X[y, x] + search_win_sizes[1] + win_sizes[1]
                    j = Y[y, x] + search_win_sizes[0] + win_sizes[0]

                    iw1 = img1[j - win_sizes[0]:j + win_sizes[0], i - win_sizes[1]:i + win_sizes[1]]
                    iw2 = img2[j - win_sizes[0] - search_win_sizes[0]:j + win_sizes[0] + search_win_sizes[0],
                          i - win_sizes[1] - search_win_sizes[1]:i + win_sizes[1] + search_win_sizes[1]]

                    cc = cv2.matchTemplate(iw2, iw1, cv2.TM_CCOEFF_NORMED)
                    cc_max = np.max(cc)

        return X, Y, dX, dY, flag

    @staticmethod
    def post(dX, dY, flag):
        def universal_outlier_detection(dx, dy, _flag, size=1):
            n_y, n_x = dx.shape
            epsilon = 0.1

            for m in range(0, 10, 1):
                for y in range(size, n_y - size, 1):
                    for x in range(size, n_x - size, 1):
                        if _flag[y, x] == 1:
                            continue

                        dx_sub = dx[y - size:y + size + 1, x - size:x + size + 1]
                        dy_sub = dy[y - size:y + size + 1, x - size:x + size + 1]
                        flag_sub = _flag[y - size:y + size + 1, x - size:x + size + 1]

                        mask = np.ones((2 * size + 1, 2 * size + 1))
                        mask[flag_sub == 0] = np.nan
                        mask[size, size] = np.nan

                        dx_sub = dx_sub * mask
                        dy_sub = dy_sub * mask
                        flag_sub = flag_sub * mask

                        if np.nansum(flag_sub) >= 3:
                            dx_med = np.nanmedian(dx_sub)
                            dy_med = np.nanmedian(dy_sub)
                            rmX = np.nanmedian(np.abs(dx_sub - dx_med))
                            rmY = np.nanmedian(np.abs(dy_sub - dy_med))
                            r0sX = np.abs(dx[y, x] - dx_med) / (rmX + epsilon)
                            r0sY = np.abs(dy[y, x] - dy_med) / (rmY + epsilon)
                            r0s = np.sqrt(r0sX**2 + r0sY**2)

                            if r0s > 2:
                                dx[y, x] = np.nanmedian(dx_sub)
                                dy[y, x] = np.nanmedian(dy_sub)
                                _flag[y, x] = 1

            return dx, dy, _flag

        def quantile_outlier_detection(dx, dy, _flag, factor=1.5):
            q1_x = np.quantile(dx, 0.25)
            q1_y = np.quantile(dy, 0.25)
            q3_x = np.quantile(dx, 0.75)
            q3_y = np.quantile(dy, 0.75)

            in_qr_x = q3_x - q1_x
            in_qr_y = q3_y - q1_y

            low_x = q1_x - factor * in_qr_x
            high_x = q3_x + factor * in_qr_x
            low_y = q1_y - factor * in_qr_y
            high_y = q3_y + factor * in_qr_y

            idx = np.where((dx < low_x) | (dx > high_x) | (dy < low_y) | (dy > high_y))
            _flag[idx] = 1

            return _flag

        def interpolation(df, _flag):
            n_y, n_x = df.shape

            for m in range(0, 100, 1):
                error_max = 0
                idx_0 = np.where(_flag == 1)

                n_0 = len(idx_0[0][:])

                for n in range(n_0):
                    x = idx_0[1][n]
                    y = idx_0[0][n]

                    if x == 0 or x == n_x - 1 or y == 0 or y == n_y - 1:
                        continue

                    elif x == 1 or x == n_x - 2 or y == 1 or y == n_y - 2:
                        df_tmp = (df[y - 1, x] + df[y + 1, x] + df[y, x - 1] + df[y, x + 1]) / 4.0

                    else:
                        df_tmp = (df[y - 1, x] + df[y + 1, x] + df[y, x - 1] + df[y, x + 1]) / 3.0 \
                                 - (df[y - 2, x] + df[y + 2, x] + df[y, x - 2] + df[y, x + 2]) / 12.0

                    error = np.abs(df[y, x] - df_tmp)

                    if error >= error_max:
                        error_max = error

                    df[y, x] = df_tmp
                    _flag[y, x] = 0

                if error_max < 0.1 and error_max != 0:
                    break

            return df, _flag

        # outlier detection
        flag = quantile_outlier_detection(dx=dX, dy=dY, _flag=flag)
        dX, dY, flag = universal_outlier_detection(dx=dX, dy=dY, _flag=flag)

        # interpolation
        dX, _ = interpolation(df=dX, _flag=flag)
        dY, _ = interpolation(df=dY, _flag=flag)

        return dX, dY

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

        def sub_pixel_analysis(ji, cc, t_size):
            res = np.zeros((1, 2))

            for j, i in zip(ji[0], ji[1]):
                if 0 < j < cc.shape[0] - 1 and 0 < i < cc.shape[1] - 1:
                    # judge whether peak value or not among 3x3
                    if cc[j, i] == np.amax(cc[j - 1:j + 2, i - 1:i + 2]):
                        eps_x = 0.5 * (np.log(cc[j, i - 1]) - np.log(cc[j, i + 1])) \
                                / (np.log(cc[j, i - 1]) + np.log(cc[j, i + 1]) - 2 * np.log(cc[j, i]))
                        eps_y = 0.5 * (np.log(cc[j - 1, i]) - np.log(cc[j + 1, i])) \
                                / (np.log(cc[j - 1, i]) + np.log(cc[j + 1, i]) - 2 * np.log(cc[j, i]))

                        # round half-up
                        eps_x = float(Decimal(str(eps_x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))
                        eps_y = float(Decimal(str(eps_y)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))

                        res = np.vstack([res, [i + eps_x + t_size[1] // 2, j + eps_y + t_size[0] // 2]])

            return np.unique(res[1:, :], axis=0)  # Exclude duplicates just in case...

        tracer_img = np.zeros((15, 15), dtype=np.uint8)
        tracer_img = gauss_circle(image=tracer_img, sd=2)

        c = cv2.matchTemplate(img, tracer_img, cv2.TM_CCORR_NORMED)
        c_j, c_i = np.where(c > threshold)

        p_xy = sub_pixel_analysis(ji=[c_j, c_i], cc=c, t_size=tracer_img.shape)

        return p_xy

    @staticmethod
    def interpolate_vectors(G_X, G_Y, G_dX, G_dY, pp):
        """
        Interpolation vectors of particle displacement
        :param G_X: [2D ndarray] x-coordinates
        :param G_Y: [2D ndarray] y-coordinates
        :param G_dX: [2D ndarray] x-component of velocity field by PIV
        :param G_dY: [2D ndarray] y-component of velocity field by PIV
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
            return np.exp(-(a ** 2 + b ** 2) / sigma ** 2)

        p_dxdy = np.zeros_like(pp)
        p = 0

        for x, y in pp:
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
                    if np.isnan(G_dX[J + j, I + i]) or np.isnan(G_dY[J + j, I + i]):
                        pass
                    else:
                        p_dxdy[p, 0] += G_dX[J + j, I + i] * w(a=X[I + i] - x, b=Y[J + j] - y, sigma=2)
                        p_dxdy[p, 1] += G_dY[J + j, I + i] * w(a=X[I + i] - x, b=Y[J + j] - y, sigma=2)
                        W += w(a=X[I + i] - x, b=Y[J + j] - y, sigma=2)

            if W == 0:
                p_dxdy[p] = np.nan
            else:
                p_dxdy[p] /= W

            p += 1

        return p_dxdy

    @staticmethod
    def nn_method(pp_a, pp_b, P_dXdY, s=5):
        class Vector2D(object):
            def __init__(self):
                self.x = 0.0
                self.y = 0.0

        class TP(object):
            def __init__(self, idx):
                self.idx = idx
                self.p0 = 0
                self.p1 = 0
                self.err = 0.0
                self.flag = False

        if pp_a is None or pp_b is None:
            return None

        pp_b_prod = pp_a + P_dXdY
        result = np.zeros([pp_a.shape[0], 4])

        dx = Vector2D
        pp01 = [TP(idx=x) for x in range(pp_a.shape[0])]
        npa = -1  # the number of available particle
        ntsp = 0  # the number for tracking the same particle

        for ii in range(pp_a.shape[0]):  # 1st
            flag = True  # init flag for the particle tracking
            c, cc = 1e10, 1e10  # arbitrary large number

            for jj in range(pp_b.shape[0]):  # 2nd
                dx.x = pp_b[jj][0] - pp_b_prod[ii][0]
                dx.y = pp_b[jj][1] - pp_b_prod[ii][1]

                cc = np.sqrt(dx.x ** 2 + dx.y ** 2)

                if cc <= s and cc < c:
                    c = cc

                    if flag:
                        npa += 1

                    pp01[npa].p0 = ii
                    pp01[npa].p1 = jj
                    pp01[npa].err = c
                    pp01[npa].flag = True
                    flag = False

        # post-processing
        # If different particles track the same particle,
        # giving an error flag to particle which has larger error.
        for ii in range(npa):
            if pp01[ii].flag:
                for jj in range(ii + 1, npa):
                    if pp01[ii].p1 == pp01[jj].p1:
                        if pp01[ii].err > pp01[jj].err:
                            pp01[ii].flag = False
                            ntsp += 1
                            break
                        else:
                            pp01[jj].flag = False
                            ntsp += 1

        for ii in range(npa + 1):
            if pp01[ii].flag:
                result[ii][0] = pp_a[pp01[ii].p0][0]
                result[ii][1] = pp_a[pp01[ii].p0][1]
                result[ii][2] = pp_b[pp01[ii].p1][0] - pp_a[pp01[ii].p0][0]
                result[ii][3] = pp_b[pp01[ii].p1][1] - pp_a[pp01[ii].p0][1]

        if npa == 0:
            return result

        return result[:npa]


if __name__ == '__main__':
    kc = KC(img_path="../data/test/in/", out_path="../data/test/out/", filename="C001H001S0001", fmt="bmp", n=2)
    kc.run()

    print(0)
