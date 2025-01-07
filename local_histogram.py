import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('C:\\Users\\ishik\\Downloads\\dipprojects\\lena.jpeg', cv2.IMREAD_GRAYSCALE)
r, c = image.shape

h = np.zeros(256, dtype=int)
for x in range(r):
    for y in range(c):
        h[image[x, y]] += 1

s = h / (image.shape[0] * image.shape[1])
cdf = np.cumsum(s) * 255

m1 = 255 / (r * c - 1)
result = np.zeros_like(image, dtype=np.uint8)
region_size = int(r/3)

for x1 in range(0, r, region_size):
    for y1 in range(0, c, region_size):

        local_region = image[x1:x1+region_size, y1:y1+region_size]

        h1 = np.zeros(256, dtype=int)
        for i1 in range(len(local_region)):
            for j1 in range(len(local_region)):
                h1[local_region[i1, j1]] += 1

        s1 = h1/ (local_region.shape[0]*local_region.shape[1])
        cdf1 = np.cumsum(s1)*255

        local_region_equalized = np.interp(local_region.flatten(), range(256), cdf1).reshape(local_region.shape).astype(np.uint8)

        result[x1:x1+region_size, y1:y1+region_size] = local_region_equalized

plt.subplot(2, 1, 1)
plt.imshow(image, cmap='gray')

plt.subplot(2, 1, 2)
plt.imshow(result, cmap='gray')

