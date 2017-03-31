import numpy as np
import cv2
from DrawMatches import drawMatches

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

# find and compute the descriptors with ORB
FixKp, FixDes = orb.detectAndCompute(FixIm, None)
# draw only keypoints location,not size and orientation
FixImKp = cv2.drawKeypoints(FixIm, FixKp, None, color=(0,255,0), flags=0)

# compute the descriptors with ORB
MovKp, MovDes = orb.detectAndCompute(MovIm, None)
# draw only keypoints location,not size and orientation
MovImKp = cv2.drawKeypoints(MovIm, MovKp, None, color=(0,255,0), flags=0)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(FixDes, MovDes)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
matches_im = drawMatches(FixIm, FixKp, MovIm, MovKp, matches[:10])


# write the result images to file
cv2.imwrite('FixIm.tif', FixIm)
cv2.imwrite('FixImKp.tif', FixImKp)
cv2.imwrite('MovIm.tif', MovIm)
cv2.imwrite('MovImKp.tif', MovImKp)
cv2.imwrite('matches.tif', matches_im)
