import argparse

from os import listdir, chdir
from os.path import isfile, join

import tensorflow as tf

from cat_vs_dog import settings
from cat_vs_dog.utils import get_img_array_from_file, ImageCategory, is_valid_image

parser = argparse.ArgumentParser(description="Cats and docs recogniser.")
parser.add_argument(
    "--path", help=f"path to folder with images in folder {settings.BASE_DIR} (default = {settings.PATH})",
    default=settings.PATH
)
parser.add_argument(
    "--model-path", help=f"path to folder with model (default = {settings.MODEL_PATH})",
    default=settings.MODEL_PATH
)


def run(path: str, model_path: str):
    model = tf.keras.models.load_model(model_path)
    path = join(settings.BASE_DIR, path)
    for file in listdir(path):
        image_category = ImageCategory.unsupported
        file = join(path, file)
        if isfile(file) and is_valid_image(file):
            img_array = get_img_array_from_file(file=file)
            image_category = ImageCategory.get_from_img_array_and_model(img_array=img_array, model=model)

        print(f"{file} | {image_category.value}")


if __name__ == "__main__":
    chdir(settings.BASE_DIR)
    args = parser.parse_args()
    run(path=args.path, model_path=args.model_path)
