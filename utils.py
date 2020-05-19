import os

import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image

matplotlib.use("Agg")


def create_dir(d):
    if not tf.gfile.IsDirectory(d):
        tf.gfile.MakeDirs(d)


class File(tf.gfile.GFile):
    """Wrapper on GFile extending seek, to support what python file supports."""

    def __init__(self, *args):
        super(File, self).__init__(*args)

    def seek(self, position, whence=0):
        if whence == 1:
            position += self.tell()
        elif whence == 2:
            position += self.size()
        else:
            assert whence == 0
        super(File, self).seek(position)


def o_gfile(filename, mode):
    """Wrapper around file open, using gfile underneath.

    filename can be a string or a tuple/list, in which case the components are
    joined to form a full path.
    """
    if isinstance(filename, tuple) or isinstance(filename, list):
        filename = os.path.join(*filename)
    return File(filename, mode)


def get_batch_size(inputs):
    return tf.cast(tf.shape(inputs)[0], tf.float32)


def save_image_array(img_array, fname):
    channels = img_array.shape[-1]
    resolution = img_array.shape[2]
    img_rows = img_array.shape[0]
    img_cols = img_array.shape[1]

    img = np.full([resolution * img_rows, resolution * img_cols, channels], 0.0)
    for r in range(img_rows):
        for c in range(img_cols):
            img[(resolution * r): (resolution * (r + 1)), (resolution * (c % 10)): (resolution * ((c % 10) + 1)), :] = \
                img_array[r, c]

    img = np.flip(img, axis=-1)
    img = (img * 255).astype(np.uint8)
    # img = (img * 255.0).astype(np.unit8)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    elif img.shape[0] == 3:
        img = np.rollaxis(img, 0, 3)

    Image.fromarray(img).save(fname)
