import cv2 as cv
import numpy as np,sys
from PIL import Image
def webp_to_np(webp_image):
    rgb_image = webp_image.convert('RGB')
    numpy_array = np.array(rgb_image)
    return numpy_array
def pyramid_blending(A,B):
    # A = cv.imread('sd1.webp')
    # B = cv.imread('sd2.webp')
    assert A is not None, "file could not be read, check with os.path.exists()"
    assert B is not None, "file could not be read, check with os.path.exists()"
    B = cv.resize(B, (B.shape[1], A.shape[0]))
    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        gpA.append(G)
    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        gpB.append(G)
    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in range(5,0,-1):
        GE = cv.pyrUp(gpA[i])
        L = cv.subtract(gpA[i-1],GE)
        lpA.append(L)
    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in range(5,0,-1):
        GE = cv.pyrUp(gpB[i])
        L = cv.subtract(gpB[i-1],GE)
        lpB.append(L)
    # Now add left and right halves of images in each level
    LS = []
    for la,lb in zip(lpA,lpB):
        rows,cols,dpt = la.shape
        ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
        LS.append(ls)
    # now reconstruct
    ls_ = LS[0]
    for i in range(1,6):
        ls_ = cv.pyrUp(ls_)
        ls_ = cv.add(ls_, LS[i])
    # image with direct connecting each half
    # real = np.hstack((A[:,:cols//2],B[:,cols//2:]))
    return ls_
    cv.imwrite('Pyramid_blending2.jpg',ls_)
    # cv.imwrite('Direct_blending.jpg',real)

