import argparse

from os import listdir, chdir
from os.path import isfile, join

import tensorflow as tf

from cat_vs_dog import settings
from cat_vs_dog.utils import get_image_category, get_img_array_from_file, ImageCategory, is_good_image

parser = argparse.ArgumentParser(description="Cats and docs recogniser.")
parser.add_argument('--path', help=f"Path to folder with images in folder {settings.BASE_DIR} (default = data)", default="data")


def run(path):
    model = tf.keras.models.load_model(settings.MODEL_PATH)
    path = join(settings.BASE_DIR, path)
    for file in listdir(path):
        image_category = ImageCategory.unsupported
        file = join(path, file)
        if isfile(file) and is_good_image(file):
            img_array = get_img_array_from_file(file=file)
            image_category = get_image_category(img_array=img_array, model=model)

        print(f"{file} | {image_category.value}")


if __name__ == '__main__':
    chdir(settings.BASE_DIR)
    args = parser.parse_args()
    run(path=args.path)
