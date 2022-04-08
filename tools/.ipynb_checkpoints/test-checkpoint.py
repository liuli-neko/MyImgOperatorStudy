import cv2
import numpy as np
import matplotlib.pyplot as plt

# get image path from command line
import sys
input = sys.argv[1]

# read img
img = cv2.imread(input)

# do histogram
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
# show histogram
plt.plot(hist)
plt.show()

