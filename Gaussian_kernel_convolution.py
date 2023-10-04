import cv2
import numpy as np


# input: sigma
# output: gaussian kernel with that sigma value applied (2d np array)
def Gaussian(m, n, sigma):
    gaussian = np.zeros((m, n))
    m = m//2
    n = n//2
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = sigma * (2*np.pi) ** 2
            x2 = np.exp(-(x**2 + y**2)/(2*sigma**2))
            gaussian[x+m, y+n] = (1/x1) * x2
    return gaussian


# convolution
# input:
#   image: normalized image (np 3d array)
#   kernel: kernel for convolution
# output: kernel applied image (np 3d array)
def applyGaussian(image, kernel):

    r = image.shape[0]
    c = image.shape[1]
    gaussianImage = np.zeros((image.shape))

    for d in range(image.shape[2]):
        layer = image[:, :, d]
        m, n = kernel.shape
        w = m // 2
        h = n // 2

        paddedimg = np.zeros(((r + 2*w), (c + 2*h)))
        paddedimg[w:w+r, h:h+c] = layer

        for row in range(w, w+r):
            for col in range(h, h+c):
                sum = 0
                for i in range(m):
                    for j in range(n):
                        sum = sum + (kernel[i, j] * paddedimg[row-w+i, col-h+j])
                gaussianImage[row-w, col-h, d] = sum
    return gaussianImage.astype(np.uint8)

img1 = cv2.imread('img1.jpg')
img1 = cv2.resize(img1, (200, 200), interpolation=cv2.INTER_AREA)
kernel = Gaussian(3, 3, 1)
gauss_applied = applyGaussian(img1, kernel)
cv2.imshow('original', img1)
cv2.imshow('gaussian applied', gauss_applied)
cv2.waitKey(0)
cv2.destroyAllWindows()



