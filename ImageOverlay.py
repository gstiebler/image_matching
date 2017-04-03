import numpy as np
import cv2
import math
from DrawMatches import drawMatches

BAR_LEN_FACTOR = 0.06
NUM_SELECTED_KP = 10

imgs_folder = '../Images/'

MovIm = cv2.imread(imgs_folder + '001_CBS_010.jpg', 0)
FixIm = cv2.imread(imgs_folder + '20141014-113042_1.jpg', 0)

# Invert the image
#FixIm = 255 - FixIm

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
# FixIm = cv2.GaussianBlur(FixIm, (41, 41), 10)

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
sorted_matches = sorted(matches, key = lambda x:x.distance)

# get the best matches
selected_matches = sorted_matches[:NUM_SELECTED_KP]

# get keypoints for selected matches
selected_fix_kp = [list(FixKp[m.queryIdx].pt) for m in selected_matches]
selected_mov_kp = [list(MovKp[m.trainIdx].pt) for m in selected_matches]

selected_fix_kp = np.array([selected_fix_kp]).astype(np.int)
selected_mov_kp = np.array([selected_mov_kp]).astype(np.int)

# Draw the selected matches
matches_im = drawMatches(FixIm, FixKp, MovIm, MovKp, selected_matches)

# testing detecting the matching points automatically
mat = cv2.estimateRigidTransform(selected_fix_kp, selected_mov_kp, True)

rows, cols = FixIm.shape
warped_fix_im = cv2.warpAffine(FixIm, mat, (cols,rows))

merged_im = cv2.addWeighted(MovIm, 0.5, warped_fix_im, 0.5, 0.0)

print mat

# getting the translation, scale and rotation components from the matrix
tx = mat[0, 2]
ty = mat[1, 2]

a = mat[0, 0]
b = mat[0, 1]
c = mat[1, 0]
d = mat[1, 1]

sx = math.sqrt(math.pow(a, 2) + math.pow(c, 2))
sy = math.sqrt(math.pow(b, 2) + math.pow(d, 2))

t1 = math.atan(c / d)
t2 = math.atan(-b / a)

print "translation: %f, %f" % (tx, ty)
print "scale: %f, %f" % (sx, sy)
print "rotation: %f and %f" % (t1, t2)

# write the result images to file
cv2.imwrite('FixIm.tif', FixIm)
cv2.imwrite('FixImKp.tif', FixImKp)
cv2.imwrite('MovIm.tif', MovIm)
cv2.imwrite('MovImKp.tif', MovImKp)
cv2.imwrite('matches.tif', matches_im)
cv2.imwrite('warped.tif', warped_fix_im)
cv2.imwrite('merged.tif', merged_im)