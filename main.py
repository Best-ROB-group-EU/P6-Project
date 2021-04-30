import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
#import scipy.misc

depth_img = np.load('colour.npy')
img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)
canny = cv2.Canny(img, 100, 200)
canny_RGB = cv2.cvtColor(canny, cv2.COLOR_BGR2RGB)
cv2.putText(canny_RGB, "Threshold: (100, 200)", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40,40,255),2)
cv2.imshow("Depth_img", canny_RGB)

""" col_array = np.load("colour.npy")
depth_array = np.load("depth.npy")

plt.figure("Depth")
plt.imshow(depth_array, cmap='gray')
plt.figure("Colour")
plt.imshow(col_array, cmap='gray')
plt.show() """

cv2.waitKey(0)