# Import the required modules
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
import numpy as np
import os
import argparse as ap
from nms import nms
from config import *
import time
def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0) 
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
def test_classifer(path, extension):
    labels = []
    data = []
    pred = []
    pos_path = os.path.join(path, 'pos')
    neg_path = os.path.join(path, 'neg')
    clf = joblib.load(model_path)
    pos_num = 0
    neg_num = 0
    for img in os.listdir(pos_path):
        img_path = os.path.join(pos_path, img)
        if os.path.splitext(img_path)[1] == extension:
            data.append(img_path)
            labels.append(1)
            pos_num += 1
    for img in os.listdir(neg_path):
        img_path = os.path.join(neg_path, img)
        if os.path.splitext(img_path)[1] == extension:
            data.append(img_path)
            labels.append(0)
            neg_num += 1
    start = time.time()
    for index, img in enumerate(data):
            im = imread(img, as_grey=True)
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualise=visualize, transform_sqrt=transform_sqrt)
            fd = fd.reshape(1, -1)
            pred.extend(clf.predict(fd) == labels[index])
    end = time.time()
    print end-start
    print len(labels)
    print pred
    return pred, pos_num, neg_num, data


if __name__ == "__main__":
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-i', "--image", help="Path to the test image", required=True)
    parser.add_argument('-d','--downscale', help="Downscale ratio", default=1000,
            type=float)
    parser.add_argument('-t', "--test", help="if used for test", action="store_true")
    parser.add_argument('-v', '--visualize', help="Visualize the sliding window",
            action="store_true")
    args = vars(parser.parse_args())

    if args['test']:
        pred, pos_num, neg_num, _ = test_classifer(args["image"], '.bmp')
        num_correct = np.sum(pred)
        pos_accuracy = float(num_correct[:pos_num]) / pos_num
        neg_accuracy = float(num_correct[pos_num:pos_num+neg_num]) / neg_num
        total_accuracy = float(num_correct) / (neg_num+pos_num)
        print 'Test accuracy: pos-{} neg-{} total-{}'.format(pos_accuracy, neg_accuracy, total_accuracy)
    else:
        # Read the image
        im_src = imread(args["image"])
        im = imread(args["image"], as_grey=True)
        start = time.time()
        # min_wdw_sz = (100, 40)
        # step_size = (10, 10)
        downscale = args['downscale']
        visualize_det = args['visualize']

        # Load the classifier
        clf = joblib.load(model_path)

        # List to store the detections
        detections = []
        curr_detections = []
        # The current scale of the image
        scale = 0
        # Downscale the image and iterate
        for im_scaled in pyramid_gaussian(im, downscale=downscale):
            # This list contains detections at the current scale
            cd = []
            # If the width or height of the scaled image is less than
            # the width or height of the window, then end the iterations.
            if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0] or min_wdw_sz[1]*downscale**scale > max_wdw_sz[1]:
                break
            for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
                if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                    continue
                # Calculate the HOG features
                fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualise=visualize, transform_sqrt=transform_sqrt)
                fd = fd.reshape(1, -1)
                pred = clf.predict(fd)

                if pred == 1:
                    # print "Detection:: Location -> ({}, {})".format(x, y)
                    # print "Scale ->  {} | Confidence Score {} \n".format(scale,clf.decision_function(fd))
                    curr_detections.append((int(round(x*(downscale**scale))), int(round(y*(downscale**scale))), clf.decision_function(fd),
                        int(min_wdw_sz[0]*(downscale**scale)),
                        int(min_wdw_sz[1]*(downscale**scale)),
                        x, y))
                    cd.append(curr_detections[-1])
                # If visualize is set to true, display the working
                # of the sliding window
                if visualize_det:
                    clone = im_scaled.copy()
                    for _, _, _, _, _, x1, y1  in cd:
                        # Draw the detections at this scale
                        cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                            im_window.shape[0]), (0, 0, 0), thickness=2)
                    cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                        im_window.shape[0]), (255, 255, 255), thickness=2)
                    cv2.imshow("Sliding Window in Progress", clone)
                    cv2.waitKey(1)
            # Move the the next scale
            # curr_detections = nms(curr_detections, threshold)
            detections += curr_detections
            scale+=1
        end = time.time()
        print 'raw test time: %.2f' % (end-start)
        # Display the results before performing NMS
        clone = im_src.copy()
        for (x_tl, y_tl, _, w, h, _, _) in detections:
            # Draw the detections
            cv2.rectangle(im, (x_tl, y_tl), (x_tl+w, y_tl+h), (255, 0, 0), thickness=2)
        cv2.imshow("Raw Detections before NMS", im)
        cv2.waitKey()
        # Perform Non Maxima Suppression
        start = time.time()
        detections = nms(detections, threshold)
        end = time.time()
        print 'NMS time: %f' % (end-start)
        # Display the results after performing NMS
        for cnt, (x_tl, y_tl, score, w, h, _, _) in enumerate(detections):
            # Draw the detections
            loc_x = int(round(x_tl+w/2))
            loc_y = int(round(y_tl+h/2))
            font = cv2.FONT_HERSHEY_COMPLEX
            text = 'No.{}'.format(cnt+1)
            cv2.putText(clone, text, (loc_x+10, loc_y-10), font, .5, (0,255,0), 1)
            cv2.circle(clone, (loc_x, loc_y), 10, (0, 255, 0), 2)
            cv2.imshow("Final Detections after applying NMS", clone)
            # cv2.waitKey(500)
            # cv2.rectangle(clone, (x_tl, y_tl), (x_tl+w,y_tl+h), (0, 0, 255), thickness=2)
        cv2.imshow("Final Detections after applying NMS", clone)
        cv2.waitKey()