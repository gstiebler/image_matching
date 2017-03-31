import numpy as np
import cv2

BAR_LEN_FACTOR = 0.06

imgs_folder = '../Images/'

MovIm = cv2.imread(imgs_folder + 'g_009.tif', 0)
FixIm = cv2.imread(imgs_folder + 'Slice69.tif', 0)

# Invert the image
FixIm = 255 - FixIm

FixImHeight = FixIm.shape[0]
# calculate the height of the bar
barLen = int(FixImHeight * BAR_LEN_FACTOR)

# remove the bar on both images
FixIm = FixIm[:-barLen, :]
MovIm = MovIm[:-barLen, :]

# equalize histogram
FixIm = cv2.equalizeHist(FixIm)
MovIm = cv2.equalizeHist(MovIm)

# blur (window size must be odd)
FixIm = cv2.GaussianBlur(FixIm, (41, 41), 10)

# TODO resize?

# With sift, not available on default OpenCV installation
# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(FixIm, None)
# FixImKp = cv2.drawKeypoints(FixIm, kp)

# Initiate ORB detector
orb = cv2.ORB()

# find the keypoints with ORB
FixKp = orb.detect(FixIm, None)
# compute the descriptors with ORB
FixKp, des = orb.compute(FixIm, FixKp)
# draw only keypoints location,not size and orientation
FixImKp = cv2.drawKeypoints(FixIm, FixKp, None, color=(0,255,0), flags=0)

# find the keypoints with ORB
MovKp = orb.detect(MovIm, None)
# compute the descriptors with ORB
MovKp, des = orb.compute(MovIm, MovKp)
# draw only keypoints location,not size and orientation
MovImKp = cv2.drawKeypoints(MovIm, MovKp, None, color=(0,255,0), flags=0)


# write the result images to file
cv2.imwrite('FixIm.tif', FixIm)
cv2.imwrite('FixImKp.tif', FixImKp)
cv2.imwrite('MovIm.tif', MovIm)
cv2.imwrite('MovImKp.tif', MovImKp)
