# Dogs Cats Classifier <!-- omit in toc -->

Create an algorithm to distinguish dogs from cats

# Contents <!-- omit in toc -->

- [Setup](#setup)
    - [Download dataset](#download-dataset)
    - [Install packages](#install-packages)
- [Train models](#train-models)
- [Evaluate models](#evaluate-models)
- [Inference](#inference)
- [Project Organization](#project-organization)

# Setup

## Download dataset

1. Download [Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats/data) dataset from kaggle.
2. Unzip dataset and put `train`, `test1` to `datasets/final`.

```
─── datasets
    ├── raw
    │   └── dogs-vs-cats.zip
    └── final
        ├── train
        └── test1
```

## Install packages

```commandline
pip install -r requirements.txt
```

# Train models

Training different model type and setting.

- Training by default setting (resnet_50)

```commandline
python train.py -r "datasets/final/train" 
```

- Training with different model types

```commandline
python train.py -r "datasets/final/train" --model-type resnext50_32x4d
```

Support model types:

- ResNet: resnet18, resnet34, resnet_50, resnet_101
- ResNext: resnext50_32x4d, resnext101_32x8d
- Swin: swin_t, swin_s, swin_b

After training, the model weight will export to `model_weights/<model-type>_<exp_time>`. 

# Evaluate models

After evaluating, the results were exported to `reports/figures`.

```commandline
python evaluate.py -r "datasets/final/train" --model-path "model_weights/<model-type>_<exp_time>/model.pt"
```

# Inference

Inference a single image or images of the folder.

- Single image inference

```commandline
python test.py --image-path="datasets/final/test1/1.jpg" --model-path="model_weights/<model-type>_<exp_time>/model.pt"
```

Save `result.png` to `reports` by default.

- Images of the folder inference

```commandline
python test.py --image-folder="datasets/final/test1" --model-path="model_weights/<model-type>_<exp_time>/model.pt"
``` 

Save `results.csv` to `reports` by default.

# Project Organization

```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── datasets
│   ├── final          <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── model_weights      <- Trained and serialized models, model predictions, or model summaries
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
│
├── train.py           <- Scripts to train models
│
├── evaluate.py        <- Scripts to evaluate models
│
├── test.py            <- Scripts to predict single sample via trained models
│
└── dogs_cats_classifier                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes dogs_cats_classifier a Python module
    │
    ├── data           <- Scripts to download or generate data
    │
    ├── models         <- Scripts to construct model modules and architecture
    │ 
    └── utils          <- Scripts to help train/test pipeline

```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
