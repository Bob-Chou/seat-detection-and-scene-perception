'''
Preprocessing the training data
cut raw images into positive or negative pieces and save them perspectively
'''

import cv2
import os
import random as rd

global raw_img, raw_img_name
global point1, point2
global cut_number

cut_shape = (128, 64)
raw_shape = (1421, 800)
raw_data_path = '../data/test/raw'
raw_data_type = '.png'

def on_mouse_pos(event, x, y, flags, param):
	global raw_img, point1, point2, raw_img_name, cut_number
	img2 = raw_img.copy()
	if event == cv2.EVENT_LBUTTONDOWN:
		point1 = (x,y)
		cv2.circle(img2, point1, 10, (0, 255, 0), 1)
		cv2.imshow(raw_img_name, img2)
	elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
		cv2.rectangle(img2, point1, (x,y), (255, 0, 0), 1)
		cv2.imshow(raw_img_name, img2)
	elif event == cv2.EVENT_LBUTTONUP:
		point2 = (x,y)
		cv2.rectangle(img2, point1, point2, (0, 0, 255), 1) 
		cv2.imshow(raw_img_name, img2)
	elif event == cv2.EVENT_RBUTTONDOWN:
		min_x = min(point1[0], point2[0])     
		min_y = min(point1[1], point2[1])
		width = abs(point1[0] - point2[0])
		height = abs(point1[1] -point2[1])
		cut_img = raw_img[min_y:min_y+height, min_x:min_x+width]
		cv2.imwrite('../data/test/pos/test-{:0>4}.bmp'.format(cut_number), cv2.resize(cut_img, cut_shape))
		cut_number += 1

def on_mouse_neg(event, x, y, flags, param):
	global raw_img, point1, point2, raw_img_name, cut_number
	img2 = raw_img.copy()
	if event == cv2.EVENT_LBUTTONDOWN:
		point1 = (rd.randint(1, raw_shape[0]-cut_shape[0]), rd.randint(1, raw_shape[1]-cut_shape[1]))
		point2 = (point1[0]+cut_shape[0], point1[1]+cut_shape[1])
		cv2.rectangle(img2, point1, point2, (255, 0, 0), 5)
		cv2.imshow(raw_img_name, img2)
	elif event == cv2.EVENT_RBUTTONDOWN:
		cut_img = raw_img[point1[1]:point2[1], point1[0]:point2[0]]
		cv2.imwrite('../data/test/neg/test-{:0>4}.bmp'.format(cut_number), cv2.resize(cut_img, cut_shape))
		cut_number += 1

def get_img_list(path, extension):
	img_list = []
	for img in os.listdir(path):
		img_path = os.path.join(path, img)
		if os.path.splitext(img_path)[1] == extension:
			img_list.append(img_path)
	return img_list

def main():
	global raw_img, raw_img_name, cut_number
	raw_img_list = get_img_list(raw_data_path, raw_data_type)
	cut_number = 0
	for cnt, raw_img_name in enumerate(raw_img_list):
		print('processing {0} : {1}/{2}'.format(raw_img_name, cnt+1, len(raw_img_list)))
		raw_img = cv2.resize(cv2.imread(raw_img_name), raw_shape)
		cv2.namedWindow(raw_img_name)
		cv2.setMouseCallback(raw_img_name, on_mouse_neg)
		cv2.imshow(raw_img_name, raw_img)
		cv2.waitKey(0)
		cv2.destroyWindow(raw_img_name)

if __name__ == '__main__':
	main()