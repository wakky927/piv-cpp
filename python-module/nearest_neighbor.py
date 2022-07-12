import numpy as np


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


def nearest(pp, s=5):
    pp0, pp1 = pp[0], pp[1]

    if pp0 is None or pp1 is None:
        return None

    result = np.zeros([pp0.shape[0], 4])

    dx = Vector2D
    pp01 = [TP(idx=x) for x in range(pp0.shape[0])]
    npa = -1  # the number of available particle
    ntsp = 0  # the number for tracking the same particle

    for ii in range(pp0.shape[0]):  # 1st
        flag = True  # init flag for the particle tracking
        c, cc = 1e10, 1e10  # arbitrary large number

        for jj in range(pp1.shape[0]):  # 2nd
            dx.x = pp1[jj][0] - pp0[ii][0]
            dx.y = pp1[jj][1] - pp0[ii][1]

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

    for ii in range(npa+1):
        if pp01[ii].flag:
            result[ii][0] = pp0[pp01[ii].p0][0]
            result[ii][1] = pp0[pp01[ii].p0][1]
            result[ii][2] = pp1[pp01[ii].p1][0] - pp0[pp01[ii].p0][0]
            result[ii][3] = pp1[pp01[ii].p1][1] - pp0[pp01[ii].p0][1]

    if npa == 0:
        return result

    return result[:npa]
