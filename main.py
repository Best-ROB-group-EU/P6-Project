import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.misc

depth_img = np.load('colour.npy')
img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)
cv2.imshow("Depth_img", depth_img)

""" col_array = np.load("colour.npy")
depth_array = np.load("depth.npy")

plt.figure("Depth")
plt.imshow(depth_array, cmap='gray')
plt.figure("Colour")
plt.imshow(col_array, cmap='gray')
plt.show() """

cv2.waitKey(0)