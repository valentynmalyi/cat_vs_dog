# cat_vs_dog

## Environment setup:

Use Python 3.8

Install required python packages:

```shell
pip install -r requirements.txt
```

## Command line scripts:

### Train the model

train model:

```shell
python scripts/train.py
```

my model you can download and put it in `cat_vs_dog/train`

https://drive.google.com/file/d/1CnOEMOsQAOl9eNIAVmpIs0uraIBB5qmj/view?usp=sharing

### Classify images from the provided directory

check images:

```shell
python scripts/classify.py -h
```

Cats and docs recogniser.

    optional arguments:
        -h, --help   show this help message and exit
        --path PATH  Path to folder with images in selected folder (default = data)

example:

```shell
python scripts/classify.py --path data
```

## Run unit tests

```
pytest
```