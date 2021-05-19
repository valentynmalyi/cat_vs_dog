import tensorflow as tf


def _rot90(img, label):
    """Rotate images by 90 degrees."""
    img = tf.image.rot90(img)
    return img, label


def extend_data(train):
    """Add rotated images"""
    train90 = train.map(_rot90)
    train180 = train90.map(_rot90)
    train270 = train180.map(_rot90)
    train = train.concatenate(train90)
    train = train.concatenate(train180)
    return train.concatenate(train270)
