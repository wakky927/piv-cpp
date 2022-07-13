import numpy as np
import cv2
from scipy import ndimage

iwdir = "C:/PIVCamp/122rpm_050/img/"
ihead = "img_"
itype = ".bmp"
owdir = "C:/PIVCamp/122rpm_050/img2/"
ohead = "img_"
otype = ".bmp"

nstart = 1
nstop = 10

R = 5  # Particleよりは大きく、Interrogation Windowよりは小さく
kernel = np.ones((2*R+1, 2*R+1))
for y in range(2*R+1):
    for x in range(2*R+1): 
        if (x-R)**2 + (y-R)**2 > R**2:
            kernel[y, x] = 0


for n in range(nstart, nstop+1, 1):
    print(n)
    iname = iwdir + ihead + "%06d" %n + itype
    img = cv2.imread(iname, 0)
    #
    MAX = ndimage.maximum_filter(img, footprint=kernel, mode="constant")
    MIN = ndimage.minimum_filter(img, footprint=kernel, mode="constant")
    MAXblur = cv2.filter2D(MAX, -1, kernel/np.sum(kernel))
    MINblur = cv2.filter2D(MIN, -1, kernel/np.sum(kernel))
    SUB = MAXblur - MINblur
    SUB[SUB==0] = 255

    imgMMF = 255 * (img - MINblur).astype("float") / SUB
    oname = owdir + ohead + "%06d" %n + otype
    cv2.imwrite(oname, imgMMF)