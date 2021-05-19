import argparse
from os import listdir
from os.path import isfile, join

import tensorflow as tf

from cat_vs_dog import settings
from cat_vs_dog.utils import get_img_array_from_file, ImageCategory, is_valid_image



def _get_parser() -> argparse.ArgumentParser:
    """Get classify parser"""
    parser = argparse.ArgumentParser(
        description="Cats and docs recogniser.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--path-data", help=f"path to folder with images",
        default=settings.PATH_DATA
    )
    parser.add_argument(
        "--path-model", help=f"path to folder with model",
        default=settings.PATH_MODEL
    )
    return parser


def run(path_data: str, path_model: str):
    model = tf.keras.models.load_model(path_model)
    for file in listdir(path_data):
        image_category = ImageCategory.unsupported
        file = join(path_data, file)
        if isfile(file) and is_valid_image(file):
            img_array = get_img_array_from_file(file=file)
            image_category = ImageCategory.get_from_img_array_and_model(img_array=img_array, model=model)

        print(f"{file} | {image_category.value}")


if __name__ == "__main__":
    classify_parser = _get_parser()
    args = classify_parser.parse_args()
    run(path_data=args.path_data, path_model=args.path_model)
