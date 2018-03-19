'''
Resize the training data
'''

import cv2
import os

src_path = '../data/pos'
src_type = '.bmp'
save_path = '../data/train/pos'
save_type = '.bmp'
new_size = (80, 40)
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
		print('augmenting {} ({}/{}), saving as {:0>4}'.format(src_name, cnt+1, 
			src_num, cnt) + save_type)
		src = cv2.imread(src_name)
		dst = cv2.resize(src, new_size)
		cv2.imwrite(save_path + '/{:0>4}'.format(cnt) + save_type, dst)

if __name__ == '__main__':
	main()
