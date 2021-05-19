from __future__ import annotations
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

    @classmethod
    def get_from_float(cls, prediction: float) -> ImageCategory:
        """Return image category from float"""
        if prediction <= 0:
            return cls.cat
        if prediction >= 1:
            return cls.dog
        return cls.unknown

    @classmethod
    def get_from_img_array_and_model(cls, img_array, model) -> ImageCategory:
        """Get image category from Numpy array and model"""
        img_resized, _ = resize_image(img_array, None)
        normalized_image, _ = normalize_image(img_resized, None)
        img_expended = np.expand_dims(normalized_image, axis=0)
        prediction = model.predict(img_expended)[0][0]
        return cls.get_from_float(prediction=prediction)


def resize_image(img, label, size=settings.SIZE):
    """Resize image to our size"""
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (size, size))
    return img, label


def normalize_image(img, label):
    """Make all picsels [0, 1]"""
    img = img / 255.0
    return img, label


def get_img_array_from_file(file: str):
    """Converts file image to a Numpy array"""
    img = load_img(file)
    return img_to_array(img)


def is_valid_image(file: str) -> bool:
    """Check file is jpg or png image by its extension"""
    return pathlib.Path(file).suffix in {".jpg", ".png", ".jpeg", ".jpe", ".jif", ".jfif", ".jfi"}
