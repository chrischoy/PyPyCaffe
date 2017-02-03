# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
#
# Modified by Christopher B. Choy
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
from config import cfg


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], max_shape[2]),
                    'float32', order='C')
    for i in xrange(num_images):
        im = ims[i]
        # TODO no indexing required
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob


def gray_im_list_to_blob(ims):
    """Convert a list of images into a network input.
    """
    blob = np.zeros((len(ims), 1, ims[0].shape[0], ims[0].shape[0]),
                    'float32', order='C')
    blob[:, 0, : :] = np.array(ims)

    return blob


def blob_to_im_list(blob):
    ims = []
    for datum in blob:
        ims.append(np.clip(datum.transpose(1,2,0) +
                    cfg.PIXEL_MEANS, 0, 255).astype(np.uint8)[:,:,::-1])
    return ims


def label_list_to_blob(labels):
    max_shape = np.array([label.shape for label in labels]).max(axis=0)
    num_images = len(labels)
    blob = np.zeros((num_images, 1, max_shape[0], max_shape[1]), 'float32', order='C')
    for i in xrange(num_images):
        label = labels[i]
        # TODO no indexing required
        blob[i, 0, 0:label.shape[0], 0:label.shape[1]] = label

    # Axis order: (batch elem, channel, height, width)
    return blob
