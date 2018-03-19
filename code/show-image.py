import random
import numpy as np
import matplotlib.pyplot as plot
from skimage.io import imread

plot_scale = 10
pos_num = 708
neg_num = 1192

def show_pos_images():
	for r in range(plot_scale):
		for c in xrange(plot_scale):
			while True:
				rand_name = '{:0>4}.bmp'.format(np.random.randint(0, high=pos_num))
				try:
					img = imread('../data/train/pos/'+rand_name)
				except IOError:
					continue
				else:
					break
			plot_index = r*plot_scale+c+1
			plot.subplot(plot_scale, plot_scale, plot_index)
			plot.imshow(img)
			plot.axis('off')
	plot.suptitle('positive samples')
	plot.show()

def show_neg_images():
	for r in range(plot_scale):
		for c in xrange(plot_scale):
			while True:
				rand_name = '{:0>4}.bmp'.format(np.random.randint(0, high=neg_num))
				try:
					img = imread('../data/train/neg/'+rand_name)
				except IOError:
					continue
				else:
					break
			plot_index = r*plot_scale+c+1
			plot.subplot(plot_scale, plot_scale, plot_index)
			plot.imshow(img)
			plot.axis('off')
	plot.suptitle('negative samples')
	plot.show()

if __name__=='__main__':
	show_pos_images()
	show_neg_images()
	pass

