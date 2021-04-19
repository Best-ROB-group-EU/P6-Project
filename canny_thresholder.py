# IMPORT LIBS
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt


# DEF Functions
def canny_annotated (src, min, max):
    canny = cv2.Canny(src, min, max)
    canny_RGB = cv2.cvtColor(canny, cv2.COLOR_BGR2RGB)
    cv2.putText(canny_RGB, "Threshold: ("+str(min)+","+str(max)+")", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40,40,255),2)
    return canny_RGB


# LOAD SOURCE
depth_img = np.load('colour.npy')
src = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)

# PREPROCCESS SOURCE
img = cv2.GaussianBlur(src, (5,5), cv2.BORDER_DEFAULT)


# SHOW SOURCE AND PREPROCCESS
cv2.imshow("PREPROCCES", np.hstack((src, img)))

# FIXED SAMPLE
fixed = canny_annotated(src, 105, 110)
cv2.imshow("Fixed", fixed)


# OUTPUT VIDEO 
OUTPUT = 0 # Set to 0 when testing and set to 1 when ready to record.
if OUTPUT == 1:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 60.0, (img.shape[1], img.shape[0]))


# CANNY THRESHOLD LOOP
for i in range (1,255):
    for x in range (1,255):

        canny_RGB = canny_annotated(img, i, x)
        cv2.imshow('Proccessing....', canny_RGB)
        if OUTPUT == 1:
            out.write(canny_RGB)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if OUTPUT == 1:
    out.release() # Remember to Rename output.avi to something else or it will be overwritten.

cv2.destroyAllWindows()
