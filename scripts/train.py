import argparse

from cat_vs_dog import settings
from cat_vs_dog.train import get_model
from cat_vs_dog.train.data import get_train_data


def _get_parser() -> argparse.ArgumentParser:
    """Get train parser"""
    parser = argparse.ArgumentParser(
        description="Cats and docs model train.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--path-model", help=f"save model to folder", default=settings.PATH_MODEL
    )
    return parser


def run(path_model: str, epochs: int = settings.EPOCHS):
    train = get_train_data()
    model = get_model()
    model.fit(train, epochs=epochs)
    model.save(path_model)


if __name__ == '__main__':
    train_parser = _get_parser()
    args = train_parser.parse_args()
    run(path_model=args.path_model)
