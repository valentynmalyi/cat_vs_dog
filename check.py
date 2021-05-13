import argparse

from os import listdir
from os.path import isfile, join

import tensorflow as tf

from utils import check_image

parser = argparse.ArgumentParser(description="Cat vs Dog")
parser.add_argument('--path', help="Path to folder with images", default="data")


def run(path):
    model = tf.keras.models.load_model("train/model")
    for file in listdir(path):
        file = join(path, file)
        if isfile(file):
            print(f"{file} | {check_image(file=file, model=model)}")


if __name__ == '__main__':
    args = parser.parse_args()
    run(path=args.path)
