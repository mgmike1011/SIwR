#!/usr/bin/env python

"""code template"""

import numpy as np
from PIL import Image
import itertools
from pgmpy.models import MarkovModel
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import Mplp
import functools
import cv2


def create_factor(var_names, var_vals, params, feats, obs):
    """
    Creates factors for given variables using exponential family and provided features.
    :param var_names: list of variable names, e.g. ['A', 'B']
    :param var_vals: list of lists of variable values, e.g. [[1, 2, 3], [3, 4]]
    :param params: list of theta parameters, one for each feature, e.g. [0.4, 5]
    :param feats: list of features (functions that accept variables and observations as arguments),
                    e.g. [feat_fun1, feat_fun2], were feat_fun1 can be defined as 'def feat_fun1(A, B, obs)'
    :param obs: observations that will be passed as the last positional argument to the features
    :return: DiscreteFactor with values computed using provided features
    """
    # shape of the values array
    f_vals_shape = [len(vals) for vals in var_vals]
    # list of values, will be reshaped later
    f_vals = []
    # for all combinations of variables values
    for vals in itertools.product(*var_vals):
        # value for current combination
        cur_f_val = 0
        # for each feature
        for fi, cur_feat in enumerate(feats):
            # value of feature multipled by parameter value
            cur_f_val += params[fi] * cur_feat(*vals, obs)
        f_vals.append(np.exp(cur_f_val))
    # reshape values array
    f_vals = np.array(f_vals)
    f_vals = f_vals.reshape(f_vals_shape)

    return DiscreteFactor(var_names, f_vals_shape, f_vals)


def main():
    # maximum size of segments grid cell
    map_size = 50
    # minimum number of pixels to consider segment
    pixels_thresh = 500
    # number of image in the database
    num_images = 70

    # counter for incorrectly classified segments
    num_incorrect = 0
    for idx in range(num_images):
        print('\n\nImage ', idx)

        # read image from disk
        image = cv2.imread('export/image_%04d.jpg' % idx)
        # read ground truth classes
        labels_gt_im = cv2.imread('export/labels_%04d.png' % idx, cv2.IMREAD_ANYDEPTH).astype(int)
        labels_gt_im[labels_gt_im == 65535] = -1
        # read a priori probabilities from classifier
        prob_im = cv2.imread('export/prob_%04d.png' % idx, cv2.IMREAD_ANYDEPTH) / 65535.0
        # read image division into segments
        segments_im = cv2.imread('export/segments_%04d.png' % idx, cv2.IMREAD_ANYDEPTH)

        # display image
        cv2.imshow('original image', image)
        cv2.waitKey(100)

        # mean R, G, B values for each segment
        mean_rgb = np.zeros([map_size, map_size, 3], dtype=float)
        # number of pixels for each segment
        num_pixels = np.zeros([map_size, map_size], dtype=np.uint)
        # a priori probabilities for each segment
        prob = np.zeros([map_size, map_size, 2], dtype=float)
        # ground truth classes for each segment
        labels_gt = -1 * np.ones([map_size, map_size], dtype=int)
        for y in range(map_size):
            for x in range(map_size):
                # segment identifier
                cur_seg = y * map_size + x
                # indices of pixels belonging to the segment
                cur_pixels = np.nonzero(segments_im == cur_seg)

                if len(cur_pixels[0]) > 0:
                    num_pixels[y, x] = len(cur_pixels[0])
                    # flip because opencv stores images as BGR by default
                    mean_rgb[y, x, :] = np.flip(np.mean(image[cur_pixels], axis=0))
                    # average results from the classifier - should not be necessary
                    prob[y, x, 0] = np.mean(prob_im[cur_pixels])
                    prob[y, x, 1] = 1.0 - prob[y, x, 0]
                    # labeling boundaries don't have to be aligned with segments boundaties
                    # count which label is the most common in this segment
                    labels_unique, count = np.unique(labels_gt_im[cur_pixels], return_counts=True)
                    labels_gt[y, x] = labels_unique[np.argmax(count)]
                    pass

        # build model
        # PUT YOUR CODE HERE
        G = MarkovModel()
        nodes = []
        for i in range(map_size):
            for j in range(map_size):
                if num_pixels[i,j] > pixels_thresh:
                    nodes.append('x_' + str(i) + '_' + str(j))
        G.add_nodes_from(nodes)

        # ------------------

        # inferred image pixels
        labels_infer = -1 * np.ones([map_size, map_size], dtype=int)

        # read results of the inference
        # PUT YOUR CODE HERE

        # ------------------

        labels_class = -1 * np.ones([map_size, map_size], dtype=int)

        for y in range(map_size - 1):
            for x in range(map_size - 1):
                if num_pixels[y, x] > pixels_thresh:
                    # label as the class with the highest probability
                    if prob[y, x, 0] >= 0.5:
                        labels_class[y, x] = 0
                    else:
                        labels_class[y, x] = 1

        # count correct pixels
        cnt_corr = np.sum(np.logical_and(labels_gt == labels_infer, labels_gt != -1))

        print('Accuracy = ', cnt_corr / np.sum(labels_gt != -1))
        num_incorrect += np.sum(labels_gt != -1) - cnt_corr

        # transfer results for segments onto image
        labels_infer_im = -1 * np.ones_like(labels_gt_im)
        labels_class_im = -1 * np.ones_like(labels_gt_im)
        for y in range(map_size - 1):
            for x in range(map_size - 1):
                if labels_infer[y, x] >= 0:
                    cur_seg = y * map_size + x
                    cur_pixels = np.nonzero(segments_im == cur_seg)

                    labels_infer_im[cur_pixels] = labels_infer[y, x]
                    labels_class_im[cur_pixels] = labels_class[y, x]

        # class 0 - green, class 1 - red in BGR
        colors = np.array([[0, 255, 0], [0, 0, 255]], dtype=np.uint8)

        image_infer = image.copy()
        image_class = image.copy()
        image_gt = image.copy()

        # color pixels according to label
        for l in range(2):
            image_infer[labels_infer_im == l] = colors[l]
            image_class[labels_class_im == l] = colors[l]
            image_gt[labels_gt_im == l] = colors[l]

        # show inferred, classified, and gt image by blending the original image with the colored one
        image_infer_vis = (0.75 * image + 0.25 * image_infer).astype(np.uint8)
        cv2.imshow('inferred', image_infer_vis)

        image_class_vis = (0.75 * image + 0.25 * image_class).astype(np.uint8)
        cv2.imshow('classified', image_class_vis)

        image_gt_vis = (0.75 * image + 0.25 * image_gt).astype(np.uint8)
        cv2.imshow('ground truth', image_gt_vis)

        # uncomment to stop after each image
        # cv2.waitKey()
        cv2.waitKey(100)

    print('Incorrectly inferred ', num_incorrect, ' segments')


if __name__ == '__main__':
    main()
