import tensorflow as tf


def rot90(img, label):
    """Rotate images by 90 degrees."""
    img = tf.image.rot90(img)
    return img, label
