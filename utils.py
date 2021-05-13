import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from settings import Settings


def resize_image(img, label):
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (Settings.SIZE, Settings.SIZE))
    img = img / 255.0
    return img, label


def rot90(img, label):
    img = tf.image.rot90(img)
    return img, label


def check_image(file, model):
    try:
        img = load_img(file)
    except:
        return "unsupported_file"
    img_array = img_to_array(img)
    img_resized, _ = resize_image(img_array, None)
    img_expended = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_expended)[0][0]
    if prediction < 0.5:
        return f"cat"
    return f"dog"
