import argparse

from cat_vs_dog import settings

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
