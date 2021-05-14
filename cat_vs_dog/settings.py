from os.path import join
from pathlib import Path

size = 224
dropout = 0.2
epochs = 1
base_dir = Path(__file__).resolve().parent.parent
model_path = join(base_dir, "cat_vs_dog", "train", "model")
shuffle = 100
batch = 128