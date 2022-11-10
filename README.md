Kit-Kars
==============================

ML project on Car Segmentation with Deloitte, for DTU Deep Learning course.

### TODO before running anything:
- run `make requirements` from `dl_kitkars` folder
- run `wandb login`, then go to the Wandb of our project, copy the key and get WANDB authorization
- download [`unet_carvana_scale0.5_epoch2.pth`](https://github.com/milesial/Pytorch-UNet/releases/tag/v3.0) in `models` folder
- download `clean_data` files (3521 files) in `data/raw/carseg_data/clean_data`

### Useful Tips
- To debug: `import pdb; pdb.set_trace()`
- Start a tmux session: `tmux new -s <someName>`
- Attach the tmux session later: `tmux attach -t <someName>` 
- Kill a tmux session: `tmux kill-session`
- Remove a local commit without any changes: `git reset HEAD~1 --soft`

### Useful Links
- About tmux: https://linuxize.com/post/getting-started-with-tmux/
- HPC GPU available: https://www.hpc.dtu.dk/?page_id=2129
- MLOps course website: https://skaftenicki.github.io/dtu_mlops/

### Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.*
    |       └── carseg_data
    |           └── clean_data
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
