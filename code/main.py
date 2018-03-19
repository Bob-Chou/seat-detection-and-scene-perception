#!/usr/bin/python
import os

# # Extract the features
# pos_path = "../data/train/pos"
# neg_path = "../data/train/neg"
# os.system("python ./extract-features.py -p {} -n {}".format(pos_path, neg_path))

# # Perform training
# pos_feat_path =  "../data/features/pos"
# neg_feat_path =  "../data/features/neg"
# os.system("python ./train-classifier.py -p {} -n {}".format(pos_feat_path, neg_feat_path))

# Perform testing 
test_im_path = "../data/test/raw/test-001.png"
# test_im_path = "../data/test/raw/Selection_013.png"
# os.system("python ./test-classifier.py -i {} -d {} --visualize".format(test_im_path,1.25))
os.system("python ./test-classifier.py -i {} -d {}".format(test_im_path, 1.25))
