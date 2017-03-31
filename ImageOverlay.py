import numpy as np
import cv2

imgs_folder = '../Images/'

MovIm = cv2.imread(imgs_folder + 'g_009.tif', 0)
FixIm = cv2.imread(imgs_folder + 'Slice69.tif', 0)

print MovIm.shape
print FixIm.shape

cv2.imshow(MovIm)

print "teste"

cv2.imwrite(MovIm, 'teste.jpg')

cv2.waitKey(0)
cv2.destroyAllWindows()
