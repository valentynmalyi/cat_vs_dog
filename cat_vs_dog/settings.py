from os.path import join
from pathlib import Path

SIZE = 224
EPOCHS = 1
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = join(BASE_DIR, "cat_vs_dog", "train", "model")
SHUFFLE = 100
BATCH = 128
PATH = "data"
