import tensorflow as tf

from .global_config import Config

def set_shapes(img, label):

    img = tf.ensure_shape(img, [*Config.INPUT_SHAPE, Config.INPUT_DIM])
    label = tf.ensure_shape(label, [1])
    return img, label