'''
Data Augmentation
'''

import cv2
import os
import random as rd

global src, src_name

src_path = '../data/test/neg'
src_type = '.bmp'

def get_img_list(path, extension):
	img_list = []
	for img in os.listdir(path):
		img_path = os.path.join(path, img)
		if os.path.splitext(img_path)[1] == extension:
			img_list.append(img_path)
	return img_list

def main():
	global src, src_name
	src_list = get_img_list(src_path, src_type)
	src_num = len(src_list)
	for cnt, src_name in enumerate(src_list):
		print('augmenting {} ({}/{}), saving as /test-{:0>4}.bmp'.format(src_name, cnt+1, src_num, src_num+cnt))
		src = cv2.imread(src_name)
		dst = cv2.flip(src, 1)
		cv2.imwrite(src_path + '/test-{:0>4}.bmp'.format(src_num+cnt), dst)

if __name__ == '__main__':
	main()