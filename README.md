# Dogs Cats Classifier <!-- omit in toc -->

Create an algorithm to distinguish dogs from cats

# Contents <!-- omit in toc -->

- [Setup](#setup)
  - [Download dataset](#download-dataset)
  - [Install packages](#install-packages)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Performance](#performance)
  - [Different Training Strategy](#different-training-strategy)
  - [Different Models](#different-models)
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

# Usage

## Training

Training different model type and setting.

```commandline
Usage: python train.py [OPTIONS]

Options:
  -r, --dataset-root PATH   The root path to dataset.  [required]
  --batch-size INTEGER      Batch size. Default: 16
  --max-epochs INTEGER      Training epochs. Default: 10
  --num-workers INTEGER     Number of workers. #CPU of this machine: 16.
                            Default: 0
  --image-size INTEGER...   The size of input image. Default: (256,256)
  --fast-dev-run            Run fast develop loop of pytorch lightning
  --seed INTEGER            Random seed of train/test split. Default: 168
  --model-type TEXT         The types of model. Default: resnet50
  --accelerator TEXT        Supports passing different accelerator types
                            ("cpu", "gpu", "tpu", "ipu", "auto") as well as
                            custom accelerator instances. Default: auto
  --devices INTEGER
  --output-path TEXT        Path to output model weight. Default:
                            model_weights
  --use-lr-scheduler        Use OneCycleLR lr scheduler
  --use-auto-augment        Use AutoAugmentPolicy
  --user-pretrained-weight  Use pretrained model
  --finetune-last-layer     Finetune last layer of model
  --help                    Show this message and exit.

```

**Examples**

- Training by default setting (resnet_50)

```commandline
python train.py -r "datasets/final/train" 
```

- Training with pretrained weight, AutoAugment, and OneCycleLR. See more details
  in `scripts/different_training_strateies.sh`

```commandline
python train.py -r "datasets/final/train" --user-pretrained-weight --finetune-last-layer --use-lr-scheduler --use-auto-augment
```

- Training with different model types. See more details in `scripts/different_models.sh`

```commandline
python train.py -r "datasets/final/train" --model-type resnext50_32x4d
```

Support model types:

- ResNet: resnet18, resnet34, resnet_50, resnet_101
- ResNext: resnext50_32x4d, resnext101_32x8d
- Swin: swin_t, swin_s, swin_b

After training, the model weight will export to `model_weights/<model-type>_<exp_time>`.
Use `tensorboard --logdir model_weights` to browse training log.

## Evaluation

After evaluating, the results were exported to `reports/figures`.

```commandline
Usage: python evaluate.py [OPTIONS]

Options:
  -r, --dataset-root PATH  The root path to dataset.  [required]
  --model-path PATH        Path to the model weight  [required]
  --batch-size INTEGER     Batch size. Default: 16
  --num-workers INTEGER    Number of workers. #CPU of this machine: 16.
                           Default: 0
  --image-size INTEGER...  The size of input image. Default: (256,256)
  --seed INTEGER           Random seed of train/test split. Default: 168
  --output-path TEXT       Path to output model weight. Default:
                           reports/figures
  --help                   Show this message and exit.

```

**Examples**

- Evaluate trained model

```commandline
python evaluate.py -r "datasets/final/train" --model-path "model_weights/<model-type>_<exp_time>/model.pt"
```

## Inference

Inference a single image or images of the folder.

```commandline
Usage: python test.py [OPTIONS]

Options:
  --image-path PATH        Path to the single image.
  --image-folder PATH      Path to the images folder
  --model-path PATH        Path to the model weight  [required]
  --image-size INTEGER...  The size of input image. Default: (256,256)
  --output-path TEXT       Path to output model prediction. Default: reports
  --batch-size INTEGER     Batch size. Default: 32
  --help                   Show this message and exit.

```

**Examples**

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

# Performance

## Different Training Strategy

Compare different training strategies performance

```commandline
bash scripts/different_training_strateies.sh
```

- model-type: resnet50
- batch-size: 16
- max-epochs: 10
- seed: 168
- image-size: (256, 256)

| Setting             | Pretrained Model | OneCycle | AutoAugment | Accuracy   |
|---------------------|------------------|----------|-------------|------------|
| From scratch        |                  |          |             | 0.8852     |
|                     |                  | V        |             | 0.9416     |
|                     |                  |          | V           | 0.8932     |
|                     |                  | V        | V           | 0.9360     |
| Train Whole Model   | V                |          |             | 0.9784     |
|                     | V                | V        |             | 0.9892     |
|                     | V                |          | V           | 0.9828     |
|                     | V                | V        | V           | 0.9920     |
| Finetune Last Layer | V                |          |             | 0.9928     |
|                     | V                | V        |             | 0.9912     |
|                     | V                |          | V           | 0.9948     |
|                     | **V**            | **V**    | **V**       | **0.9944** |

## Different Models

Compare different models performance

```commandline
bash scripts/different_models.sh
```

| Models           | Accuracy   |
|------------------|------------|
| resnet18         |     0.9844 |
| resnet34         |     0.9832 |
| resnet50         |     0.9944 |
| resnet101        |     0.9964 |
| resnext50_32x4d  |     0.9932 |
| resnext101_32x8d |     0.9944 |
| swin_t           |     0.9940 |
| swin_s           |     0.9964 |
| **swin_b**       | **0.9976** |

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
├── scripts            <- Scripts to train model in different setting
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
├── test.py            <- Scripts to predict single sample or multiple images via trained models
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
