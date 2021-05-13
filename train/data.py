import tensorflow_datasets as tfds
from utils import resize_image, rot90


def download_data():
    data_set, _ = tfds.load('cats_vs_dogs', split=['train[:100%]'], with_info=True, as_supervised=True)
    train = data_set[0]
    train = train.map(resize_image)
    # copy and turn images
    train90 = train.map(rot90)
    train180 = train90.map(rot90)
    train270 = train180.map(rot90)
    # concatenate
    train = train.concatenate(train90)
    train = train.concatenate(train180)
    train = train.concatenate(train270)
    return train.shuffle(100).batch(16)