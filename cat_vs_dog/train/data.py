import tensorflow_datasets as tfds

from cat_vs_dog import settings
from cat_vs_dog.utils import resize_image, normalize_image
from cat_vs_dog.train.utils import extend_data


def download_data():
    """Download data from tensorflow datasets. If dog label = 1, if cat label = 0"""
    data_set, _ = tfds.load('cats_vs_dogs', split=['train[:100%]'], with_info=True, as_supervised=True)
    return data_set[0]


def get_train_data():
    """Get train data"""
    train = download_data()
    train = train.map(resize_image)
    train = train.map(normalize_image)
    train = extend_data(train=train)
    return train.shuffle(settings.shuffle).batch(settings.batch)
