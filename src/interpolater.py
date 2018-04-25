# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.image

from .logger import create_logger
from .loader import load_images, DataSampler
from .utils import bool_flag

class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

def interpolate(name="",
                model_path="", 
                n_images=10, 
                offset=0, 
                n_interpolations=10, 
                alpha_min=1, 
                alpha_max=1, 
                plot_size=5,
                row_wise=True, 
                output_path="output.png") :
    # parse parameters
#     parser = argparse.ArgumentParser(description='Attributes swapping')
#     parser.add_argument("--model_path", type=str, default="",
#                         help="Trained model path")
#     parser.add_argument("--n_images", type=int, default=10,
#                         help="Number of images to modify")
#     parser.add_argument("--offset", type=int, default=0,
#                         help="First image index")
#     parser.add_argument("--n_interpolations", type=int, default=10,
#                         help="Number of interpolations per image")
#     parser.add_argument("--alpha_min", type=float, default=1,
#                         help="Min interpolation value")
#     parser.add_argument("--alpha_max", type=float, default=1,
#                         help="Max interpolation value")
#     parser.add_argument("--plot_size", type=int, default=5,
#                         help="Size of images in the grid")
#     parser.add_argument("--row_wise", type=bool_flag, default=True,
#                         help="Represent image interpolations horizontally")
#     parser.add_argument("--output_path", type=str, default="output.png",
#                         help="Output path")
#     params = parser.parse_args()
    
    params = Map()
    params.name = name
    params.model_path = model_path
    params.n_images = n_images
    params.offset = offset
    params.n_interpolations = n_interpolations
    params.alpha_min = alpha_min
    params.alpha_max = alpha_max
    params.plot_size = plot_size
    params.row_wise = row_wise
    params.output_path = output_path

    # check parameters
    assert os.path.isfile(params.model_path)
    assert params.n_images >= 1 and params.n_interpolations >= 2

    # create logger / load trained model
    logger = create_logger(None)
    ae = torch.load(params.model_path).eval()

    # restore main parameters
    params.debug = True
    params.batch_size = 32
    params.v_flip = False
    params.h_flip = False
    params.img_sz = ae.img_sz
    params.attr = ae.attr
    params.n_attr = ae.n_attr
    if not (len(params.attr) == 1 and params.n_attr == 2):
        raise Exception("The model must use a single boolean attribute only.")

    # load dataset
    data, attributes = load_images(params)
    test_data = DataSampler(data[2], attributes[2], params)


    def get_interpolations(ae, images, attributes, params):
        """
        Reconstruct images / create interpolations
        """
        assert len(images) == len(attributes)
        enc_outputs = ae.encode(images)

        # interpolation values
        alphas = np.linspace(1 - params.alpha_min, params.alpha_max, params.n_interpolations)
        alphas = [torch.FloatTensor([1 - alpha, alpha]) for alpha in alphas]

        # original image / reconstructed image / interpolations
        outputs = []
        outputs.append(images)
        outputs.append(ae.decode(enc_outputs, attributes)[-1])
        for alpha in alphas:
            alpha = Variable(alpha.unsqueeze(0).expand((len(images), 2)).cuda())
            outputs.append(ae.decode(enc_outputs, alpha)[-1])

        # return stacked images
        return torch.cat([x.unsqueeze(1) for x in outputs], 1).data.cpu()


    interpolations = []

    for k in range(0, params.n_images, 100):
        i = params.offset + k
        j = params.offset + min(params.n_images, k + 100)
        images, attributes = test_data.eval_batch(i, j)
        interpolations.append(get_interpolations(ae, images, attributes, params))

    interpolations = torch.cat(interpolations, 0)
    assert interpolations.size() == (params.n_images, 2 + params.n_interpolations,
                                     3, params.img_sz, params.img_sz)


    def get_grid(images, row_wise, plot_size=5):
        """
        Create a grid with all images.
        """
        n_images, n_columns, img_fm, img_sz, _ = images.size()
        if not row_wise:
            images = images.transpose(0, 1).contiguous()
        images = images.view(n_images * n_columns, img_fm, img_sz, img_sz)
        images.add_(1).div_(2.0)
        return make_grid(images, nrow=(n_columns if row_wise else n_images))


    # generate the grid / save it to a PNG file
    grid = get_grid(interpolations, params.row_wise, params.plot_size)
    matplotlib.image.imsave(params.output_path, grid.numpy().transpose((1, 2, 0)))
