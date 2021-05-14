import tensorflow_datasets as tfds

from cat_vs_dog import settings
from cat_vs_dog.utils import resize_image, normalize_image
from cat_vs_dog.train.utils import rot90


def download_data():
    data_set, _ = tfds.load('cats_vs_dogs', split=['train[:100%]'], with_info=True, as_supervised=True)
    return data_set[0]


def turn_data(train):
    """copy and turn images"""
    train90 = train.map(rot90)
    train180 = train90.map(rot90)
    train270 = train180.map(rot90)
    return train90, train180, train270


def concatenate_train_data(train, train90, train180, train270):
    train = train.concatenate(train90)
    train = train.concatenate(train180)
    return train.concatenate(train270)


def get_train_data():
    train = download_data()
    train = train.map(resize_image)
    train = train.map(normalize_image)
    train90, train180, train270 = turn_data(train=train)
    train = concatenate_train_data(train, train90, train180, train270)
    return train.shuffle(settings.shuffle).batch(settings.batch)
