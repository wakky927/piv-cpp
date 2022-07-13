import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from numpy.lib.function_base import meshgrid

iwdir = "C:/PIVCamp/122rpm_050/img2/"
ihead = "img_"; itype = ".bmp"
owdir = "C:/PIVCamp/"
ohead = "PIV_"; otype = ".csv"

nstart = 1
nstop = 1
nskip = 1
niter = 1

WIDTH = 1024; HEIGHT = 1024

Nx = 65; Ny = 65  # Grid Number
W = 16; H = 16  # (2W * 2H)
Sx = 10; Sy = 10  # Search Window size (W + Sx)
THRESHOLD = 0.3  # 画像のクオリティで決める


def QuantileOutlierDetection(dXorg, dYorg, Flagorg, factor=1.5):
    dX = np.copy(dXorg)
    dY = np.copy(dYorg)
    Flag = np.copy(Flagorg)
    FlagPre = np.sum(Flag==0)
    Q1x = np.quantile(dX, 0.25); Q3x = np.quantile(dX, 0.75)
    Q1y = np.quantile(dY, 0.25); Q3y = np.quantile(dY, 0.75)
    IQRx = Q3x - Q1x; IQRy = Q3y - Q1y
    QLx = Q1x - factor * IQRx; QHx = Q3x + factor * IQRx
    QLy = Q1y - factor * IQRy; QHy = Q3y + factor * IQRy
    ID = np.where((dX<QLx) | (dX>QHx) | (dY<QLy) | (dY>QHy))
    dX[ID] = 0; dY[ID] = 0; Flag[ID] = 0
    FlagPost = np.sum(Flag==0)
    NumReplaced = FlagPost - FlagPre
    print("Quantile Outlier Detection > Replaced %.2f%% Vectors" %(100*NumReplaced/(Nx*Ny)))
    return dX, dY, Flag


def UniversalOutlierDetection(dXorg, dYorg, Flagorg, size=1):
    dX = np.copy(dXorg)
    dY = np.copy(dYorg)
    Flag = np.copy(Flagorg)
    epsilon = 0.1  #subpixel accuracy
    for m in range(0, 10, 1):
        for y in range(size, Ny-size, 1):
            for x in range(size, Nx-size, 1):
                if Flag[y,x] == 0:
                    continue
                dXsub = dX[y-size:y+size+1, x-size:x+size+1]
                dYsub = dY[y-size:y+size+1, x-size:x+size+1]
                Flagsub = Flag[y-size:y+size+1, x-size:x+size+1]
                Mask = np.ones((2*size+1, 2*size+1))
                Mask[Flagsub==0] = np.nan
                Mask[size, size] = np.nan
                dXsub = dXsub * Mask
                dYsub = dYsub * Mask
                Flagsub = Flagsub * Mask
                if np.nansum(Flagsub) >= 3:
                    dXmed = np.nanmedian(dXsub)
                    dYmed = np.nanmedian(dYsub)
                    rmX = np.nanmedian(np.abs(dXsub - dXmed))
                    rmY = np.nanmedian(np.abs(dYsub - dYmed))
                    r0sX = np.abs(dX[y, x] - dXmed) / (rmX + epsilon)
                    r0sY = np.abs(dY[y, x] - dYmed) / (rmY + epsilon)
                    r0s = np.sqrt(r0sX**2 + r0sY**2)
                    if r0s > 2:
                        dX[y, x] = np.nanmedian(dXsub)
                        dY[y, x] = np.nanmedian(dYsub)
                        Flag[y,x] = 0
    return dX, dY, Flag


def CubicSplineInterpolation(dX, Flag):
    for m in range(0, 100, 1):
        errormax = 0
        ID0 = np.where(Flag==0)
        N0 = len(ID0[0][:])
        for n in range(N0):
            x = ID0[1][n]
            y = ID0[1][n]
            if x==0 or x ==Nx-1 or y==0 or y==Ny-1:
                continue
            elif x==1 or x ==Nx-2 or y==1 or y==Ny-2:
                dXs = (dX[y-1,x] + dX[y+1,x] + dX[y,x-1] + dX[y,x+1]) / 4.0
            else:
                dXs = (dX[y-1,x] + dX[y+1,x] + dX[y,x-1] + dX[y,x+1]) / 3.0 - (dX[y-2,x] + dX[y+2,x] + dX[y,x-2] + dX[y,x+2]) /12.0
            error = np.abs(dX[y,x]- dXs)
            if error >= errormax:
                errormax = error
            dX[y,x] = dXs
        if errormax < 0.1 and errormax != 0:
            break
    return dX


def DrawVector(X, Y, dX, dY, size=5):
    fig = plt.figure(figsize=(size,size*HEIGHT/WIDTH))
    ax = fig.add_axes([0,0,1,1])
    dP = np.sqrt(dX**2 + dY**2)
    dPmax = np.max(dP)
    ax.quiver(X-0.5, Y-0.5, dX/dPmax, -dY/dPmax, dP, cmap="jet", headaxislength=10, headlength=10, headwidth=4, scale=10)
    ax.invert_yaxis()
    plt.show()


def CheckStatistics(dX, dY, Flag, size=2):
    fig = plt.figure(figsize=(2*size, size))
    ax = fig.add_axes([0,0,1,1])
    ax.hist(dX[Flag==1], bins=10*(2*Sx), range=(-Sx, Sx), color="blue", alpha=0.5, label="dX")
    ax.hist(dY[Flag==1], bins=10*(2*Sy), range=(-Sy, Sy), color="red", alpha=0.5, label="dY")
    ax.set_xlabel("dX, dY [pixel/frame]")
    ax.set_ylabel("Count [-]")
    ax.legend()
    plt.show()


for n in range(nstart, nstop + 1, niter):
    iname1 = iwdir + ihead + "%06d" %n + itype
    iname2 = iwdir + ihead + "%06d" %(n + nskip) + itype
    img1org = cv2.imread(iname1, 0)
    img2org = cv2.imread(iname2, 0)
    img1 = np.zeros((HEIGHT+2*(Sy+H), WIDTH+2*(Sx+W))).astype("uint8")
    img2 = np.zeros((HEIGHT+2*(Sy+H), WIDTH+2*(Sx+W))).astype("uint8")
    img1[Sy+H:Sy+H+HEIGHT, Sx+W:Sx+W+WIDTH] = img1org
    img2[Sy+H:Sy+H+HEIGHT, Sx+W:Sx+W+WIDTH] = img2org

X, Y = meshgrid(linspace(0, WIDTH, Nx, dtype="int"),
                   linspace(0, HEIGHT, Ny, dtype="int"))
dX = np.zeros((Ny, Nx))
dY = np.zeros((Ny, Nx))
Flag = np.zeros((Ny, Nx))

for y in range(0, Ny, 1):
    for x in range(0, Nx, 1):
        i = X[y, x] + Sx + W
        j = Y[y, x] + Sy + H
        IW1 = img1[j-H:j+H, i-W:i+W]
        IW2 = img2[j-H-Sy:j+H+Sy, i-W-Sx:i+W+Sx]
        CC = cv2.matchTemplate(IW2, IW1, cv2.TM_CCOEFF_NORMED)
        CCmax = np.max(CC)
        if CCmax > THRESHOLD:
            ymax, xmax = np.unravel_index(np.argmax(CC), (2*Sy+1, 2*Sx+1))
            if xmax==0 or xmax==2*Sx or ymax==0 or ymax==2*Sy:
                xsub = 0; ysub = 0
            else:
                CCc = CC[ymax, xmax]
                CCt = CC[ymax-1, xmax]; CCb = CC[ymax+1, xmax]
                CCl = CC[ymax, xmax-1]; CCr = CC[ymax, xmax+1]
                if np.any(np.array([CCc, CCt, CCb, CCl, CCr]) <= 0):
                    xsub = 0; ysub = 0
                else:
                    xsub = (np.log(CCl) - np.log(CCr)) / (2*(np.log(CCl) + np.log(CCr) - 2*np.log(CCc)))
                    ysub = (np.log(CCt) - np.log(CCb)) / (2*(np.log(CCt) + np.log(CCb) - 2*np.log(CCc)))

            dX[y,x] = xmax + xsub - Sx
            dY[y,x] = ymax + ysub - Sy
            Flag[y,x] = 1
        else:
            Flag[y,x] = 0
        """
        plt.imshow(CC, clim=(-1, 1), cmap="bwr")
        plt.scatter(xmax, ymax)
        plt.show()
        """

dX2, dY2, Flag2 = QuantileOutlierDetection(dX, dY, Flag)
dX2, dY2, Flag2 = UniversalOutlierDetection(dX2, dY2, Flag2, size=1)
dX3 = CubicSplineInterpolation(dX2, Flag2)
dY3 = CubicSplineInterpolation(dY2, Flag2)
DrawVector(X, Y, dX, dY, size=5)
DrawVector(X, Y, dX2, dY2, size=5)
DrawVector(X, Y, dX3, dY3, size=5)
CheckStatistics(dX3, dY3, Flag2, size=2)

oname = owdir + ohead + "%06d" %n + otype
output = np.column_stack((X.flatten(), Y.flatten(), dX.flatten(), dY.flatten()))
np.savetxt(oname, output, delimiter=",", fmt="%.1f", header="X, Y, dX, dY", comments="")