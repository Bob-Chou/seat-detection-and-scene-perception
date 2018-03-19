from skimage.feature import hog
from skimage.io import imread
from skimage.io import imshow
import numpy as np
import cv2

enhancement_rate = 15 # to enhance the hog feature to be visiualized better
im_name = 'Selection_014.png'

im = imread('../data/test/raw/' + im_name, as_grey=True)
normalised_blocks, hog_image = hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8), visualise=True)
im_max, im_min = hog_image.max(), hog_image.min()
hog_image = enhancement_rate*(hog_image - im_min)/(im_max-im_min)
cv2.imshow('hog', hog_image)
cv2.waitKey()