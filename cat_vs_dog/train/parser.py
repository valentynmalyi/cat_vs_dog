import argparse

from cat_vs_dog import settings

parser = argparse.ArgumentParser(
    description="Cats and docs model train.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--path-model", help=f"save model to folder", default=settings.PATH_MODEL
)
