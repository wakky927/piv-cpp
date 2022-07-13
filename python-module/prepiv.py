import numpy as np
import cv2

iwdir = "C:/PIVCamp/122rpm_050/"
ihead = "img_"; itype = ".bmp"
owdir = "C:/PIVCamp/122rpm_050/img/"
ohead = "img_" ; otype = ".bmp"

oname1 = owdir + "backgroundMIN.bmp"
oname2 = owdir + "backgroundMEAN.bmp"

nstart  = 1
nstop   = 125
num     = nstop - nstart + 1

iname = iwdir + ihead + "%06d" %nstart + itype
img0  = cv2.imread(iname, 0)

imgs = np.zeros((img0.shape[0], img0.shape[1], num))  # create cube

for n in range(nstart, nstop+1, 1):
    iname = iwdir + ihead + "%06d" %n + itype
    gray = cv2.imread(iname, 0)
    imgs[:, :, n-nstart] = gray

imgMIN = np.min(imgs, axis=2)
imgMEAN = np.mean(imgs, axis=2)

cv2.imwrite(oname1, imgMIN)
cv2.imwrite(oname2, imgMEAN)


for n in range(nstart, nstop+1, 1):
    # imgBS = imgs[:,:,n-nstart] - imgMIN
    imgBS = imgs[:,:,n-nstart] - imgMEAN
    # imgBS += 20
    imgBS = np.clip(imgBS, 0, 255)

    imgBS = cv2.GaussianBlur(imgBS, (5,5), 0)

    oname = owdir + ohead + "%06d" %n + otype
    cv2.imwrite(oname, imgBS)