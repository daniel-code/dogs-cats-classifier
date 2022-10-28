dogs-cats-classifier
==============================

Create an algorithm to distinguish dogs from cats

Project Organization
------------

```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── datasets
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── final          <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── model_weights      <- Trained and serialized models, model predictions, or model summaries
│
├── logs               <- Training logs
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
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
