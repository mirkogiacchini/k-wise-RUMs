# Approximating a RUM from Distributions on k-slates

Code for the paper "Approximating a RUM from Distributions on k-slates" https://proceedings.mlr.press/v206/chierichetti23a.html

### dependencies and credits
The structure of the code is based on Almanza et al. code https://proceedings.mlr.press/v162/almanza22a.html <br>
The code for the discrete choice MNL model is from https://github.com/sragain/pcmc-nips <br>

dependencies: python >=3.9 standard installation, numpy, pandas, scikit-learn, matplotlib, docplex and cplex <br>

dataset names: 'sushiA', 'SFwork', 'SFshop', 'election/a5', 'election/a9', 'election/a17', 'election/a48', 'election/a81' <br>

The datasets must be placed in data/raw, in particular:
- sushi dataset: download sushi3-2016 from https://www.kamishima.net/sushi/, we need data/raw/sushi3-2016/sushi3a.5000.10.order

- election dataset: download from https://rangevoting.org/TidemanData.html or https://proceedings.mlr.press/v162/almanza22a.html, we need
    - data/raw/a5.hil
    - data/raw/a9.hil
    - data/raw/a17.hil
    - data/raw/a48.hil
    - data/raw/a81.hil

- SFshop/SFwork, download from https://github.com/sragain/pcmc-nips, we need data/raw/SFshop.csv, data/raw/SFwork.csv

### clean the datasets
python3 cleaner.py

### Fitting experiments
slate_size must be in [2,3,4,5]. If dataset_name and slate_size are not provided, all datasets and slate sizes are used.

- python3 rumrunner.py [dataset_name] [slate_size]

- python3 evaluator.py [dataset_name] [slate_size]

for discrete choice MNL:
- python3 pcmpMNL.py [dataset_name] [slate_size]

for classifier MNL:
- python3 MNL.py [dataset_name] [slate_size]

### Prediction experiments 

- python3 predictions.py [dataset_name] [slate_size]

- python3 prediction_results.py [dataset_name] [slate_size]