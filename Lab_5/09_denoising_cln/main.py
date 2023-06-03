#!/usr/bin/env python

"""code template"""

import numpy as np
from PIL import Image
import itertools
from pgmpy.models import MarkovModel
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import Mplp


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
    # read image from disk, it has to be in the same folder
    image_noise = Image.open('smile_noise_small.png')
    # display image
    image_noise.show()

    # intensity numpy array
    intensity = np.asarray(image_noise)[:, :, 0]

    # create graph and compute MAP assignment
    # TODO PUT YOUR CODE HERE
    # TODO Due to a bug in the MPLP implementation,
    # TODO add unary factors before pairwise ones!

    # ------------------

    # reading ground truth image
    image_gt = Image.open('smile_small.png')
    image_gt_np = np.asarray(image_gt)[:, :, 0]

    # inferred image pixels
    image_infer_np = np.zeros(intensity.shape, dtype=np.uint8)

    # read results of the inference
    # TODO PUT YOUR CODE HERE
    # TODO Due to a bug in the MPLP implementation,
    # TODO add unary factors before pairwise ones!

    # ------------------

    # count correct pixels
    cnt_corr = np.sum(image_infer_np == image_gt_np)

    print('Accuracy = ', cnt_corr / (image_gt_np.shape[0] * image_gt_np.shape[1]))

    # show inferred image
    image_infer = Image.fromarray(image_infer_np)
    image_infer.show()


if __name__ == '__main__':
    main()
