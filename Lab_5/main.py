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


# unary feature
def unary_feat(x, obs):
    # scale to avoid ln(0)
    obs = obs * 245.0 / 255.0 + 5.0
    if x == 0:
        val = np.log(obs / 255)
    else:
        val = np.log(1 - obs / 255)
    return val


# pairwise feature
def pairwise_feat(xi, xj, obs):
    # 1 if the same, 0 otherwise
    if xi == xj:
        val = 1
    else:
        val = 0
    return val


def main():
    # read image from disk, it has to be in the same folder
    image_noise = Image.open('09_denoising_cln/smile_noise_small.png')
    # display image
    image_noise.show()

    # intensity numpy array
    intensity = np.asarray(image_noise)[:, :, 0]

    # create variable nodes
    nodes = []
    for r in range(intensity.shape[0]):
        for c in range(intensity.shape[1]):
            nodes.append('x_' + str(r) + '_' + str(c))

    # add unary factors
    factors_u = []
    for r in range(intensity.shape[0]):
        for c in range(intensity.shape[1]):
            cur_f = create_factor(['x_' + str(r) + '_' + str(c)],
                                  [[0, 1]],
                                  [1.0],
                                  [unary_feat],
                                  intensity[r, c])
            factors_u.append(cur_f)

    # add pairwise factors
    factors_p = []
    edges_p = []
    # adding only to right and down neighbour to prevent duplicates
    for r in range(intensity.shape[0] - 1):
        for c in range(intensity.shape[1] - 1):
            cur_f_r = create_factor(['x_' + str(r) + '_' + str(c), 'x_' + str(r + 1) + '_' + str(c)],
                                    [[0, 1], [0, 1]],
                                    [0.08],
                                    [pairwise_feat],
                                    None)
            cur_f_c = create_factor(['x_' + str(r) + '_' + str(c), 'x_' + str(r) + '_' + str(c + 1)],
                                    [[0, 1], [0, 1]],
                                    [0.08],
                                    [pairwise_feat],
                                    None)
            factors_p.append(cur_f_r)
            factors_p.append(cur_f_c)
            # add edges
            edges_p.append(('x_' + str(r) + '_' + str(c), 'x_' + str(r + 1) + '_' + str(c)))
            edges_p.append(('x_' + str(r) + '_' + str(c), 'x_' + str(r) + '_' + str(c + 1)))

    # Markov model, because MPLP doesn't support factor graphs
    G = MarkovModel()
    G.add_nodes_from(nodes)
    print('Adding factors_u')
    G.add_factors(*factors_u)
    print('Adding factors_p')
    G.add_factors(*factors_p)
    print('Adding edges')
    G.add_edges_from(edges_p)

    # checking if everthing is ok
    print('Check model :', G.check_model())

    # initialize inference algorithm
    denoise_infer = Mplp(G)

    # inferring MAP assignment
    q = denoise_infer.map_query()

    # reading ground truth image
    image_gt = Image.open('09_denoising_cln/smile_small.png')
    image_gt_np = np.asarray(image_gt)[:, :, 0]

    # inferred image pixels
    image_infer_np = np.zeros(intensity.shape, dtype=np.uint8)

    # counter of all pixels
    cnt = 0
    for r in range(intensity.shape[0]):
        for c in range(intensity.shape[1]):
            # name of the variable for this pixel
            var = 'x_' + str(r) + '_' + str(c)
            # value in unary factor for this pixel
            val = q[var]
            if(val==0):
                image_infer_np[r, c] = 255
            else:
                image_infer_np[r, c] = 0


    # count correct pixels
    cnt_corr = np.sum(image_infer_np == image_gt_np)

    print('Accuracy = ', cnt_corr / (image_gt_np.shape[0] * image_gt_np.shape[1]))

    # show inferred image
    image_infer = Image.fromarray(image_infer_np)
    image_infer.show()


if __name__ == '__main__':
    main()