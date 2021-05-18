# cat vs dog

## Environment setup:

Use Python 3.8

Install required python packages:
```shell
pip install -r requirements.txt
```
## Command line scripts:
Add root folder into ```PYTHONPATH```
### Train the model
Train model:
```shell
python scripts/train.py
```
Documentation:
```shell
python scripts/train.py -h
```
`usage: train.py [-h] [--path-model PATH_MODEL]`

Cats and docs model train.

    optional arguments:
        -h, --help            show this help message and exit
        --path-model PATH_MODEL save model to folder (default: /opt/project/cat_vs_dog/train/model)
### Model
My model you can download and put it in `cat_vs_dog/train`
https://drive.google.com/file/d/1CnOEMOsQAOl9eNIAVmpIs0uraIBB5qmj/view?usp=sharing
### Classify images from the provided directory
Check images in data folder:
```shell
python scripts/classify.py
```
Documentation:
```shell
python scripts/classify.py -h
```
`usage: classify.py [-h] [--path-data PATH_DATA] [--path-model PATH_MODEL]`

Cats and docs recogniser.

    optional arguments:
        -h, --help            show this help message and exit
        --path-data PATH_DATA path to folder with images (default = /opt/project/data)
        --path-model PATH_MODEL path to folder with model (default = /opt/project/cat_vs_dog/train/model)

## Run unit tests
```
pytest
```