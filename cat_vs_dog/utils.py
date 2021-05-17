import pathlib

from enum import Enum

import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from cat_vs_dog import settings


class ImageCategory(Enum):
    """All image categories"""
    dog = "dog"
    cat = "cat"
    unknown = "unknown_class"
    unsupported = "unsupported_file"


def resize_image(img, label, size=settings.size):
    """Resize image to our size"""
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (size, size))
    return img, label


def normalize_image(img, label):
    """Make all picsels [0, 1]"""
    img = img / 255.0
    return img, label


def get_img_array_from_file(file):
    """Converts file image to a Numpy array"""
    img = load_img(file)
    return img_to_array(img)


def get_image_category(img_array, model):
    """Get image category from Numpy array and model"""
    img_resized, _ = resize_image(img_array, None)
    normalized_image, _ = normalize_image(img_resized, None)
    img_expended = np.expand_dims(normalized_image, axis=0)
    prediction = model.predict(img_expended)[0][0]
    return predict(prediction=prediction)


def predict(prediction) -> ImageCategory:
    """Return image category from float"""
    if prediction <= 0:
        return ImageCategory.cat
    if prediction >= 1:
        return ImageCategory.dog
    return ImageCategory.unknown


def is_good_image(file) -> bool:
    """Check file end with jpg or png"""
    return pathlib.Path(file).suffix in {".jpg", ".png"}
