# Filter Learning Framework (fork)

Automatic search of an image preprocessing pipeline that reduces the domain
shift between high-cost (HCM) and low-cost (LCM) microscope images for malaria
parasite detection with YOLO.

This repository is a fork of
[andreolli-davide/filter_learning_framework](https://github.com/andreolli-davide/filter_learning_framework),
developed by Davide Senette and Davide Andreolli for the *Signal, Image and
Video* course at the University of Trento. The changes described below were
made by Davide Valer for the bachelor thesis *"Machine Learning No-Code per
Immagini Biomediche: Rilevamento della Malaria con KNIME"* (University of
Trento, 2026). The version described in the thesis is tagged `tesi-v1`.

The trained models and the filter pipeline can also be used without code
through the KNIME nodes developed for the thesis: <!-- TODO: link to the KNIME nodes repository -->

## How it works

The candidate filters are organized in four layers, always applied in this
order, and each layer can also apply no filter:

| Layer | Filters |
|-------|---------|
| 1. Noise reduction | median blur, bilateral filter |
| 2. Color correction | saturation boost, gamma correction |
| 3. Contrast enhancement | CLAHE |
| 4. Sharpening | unsharp mask, Laplacian sharpening |

The pipeline is built greedily, one layer at a time. For each candidate filter
an Optuna study searches the parameters that maximize the mAP50 of the model on
a set of LCM images; the best filter is added to the pipeline and the search
moves to the next layer.

The model can also be trained with a **targeted data augmentation**: during
training, a random filter with random parameters (or no filter) is applied to
each HCM image for every layer, so that the model learns to handle images
transformed by the same filters.

## Changes with respect to the original framework

**Dataset (`src/dataset.py`)**
- The dataset class no longer depends on the M5 folder layout: domain and split
  are read from the folder names, whatever they are.
- Images are loaded only when needed and released afterwards.
- Loading from files is separated from the rest of the class, so that other
  data sources (e.g. KNIME tables) can be added.

**Filter search (`src/orchestrator.py`, `src/trainer.py`)**
- The filters chosen at previous layers are kept with their parameters
  (frozen): each study optimizes only the filter of the current layer.
- The number of trials of each study is computed automatically from the size
  of the filter's search space: `n = ceil(ln(1 - c) / ln(1 - p)) = 29` with
  `p = 0.10`, `c = 0.95`, at least 10 trials per parameter, and never more than
  the number of possible configurations (hard cap: `n_trials_per_combination`).
- Small spaces are searched exhaustively with a grid search, large ones with TPE.
- TPE studies stop early after 15 trials without an mAP50 gain of at least 0.005.
- With the default layers the search needs at most 137 trials instead of 210.

| Filter | Configurations | Trials | Method |
|--------|---------------:|-------:|--------|
| Median blur | 2 | 2 | grid |
| Gamma correction | 7 | 7 | grid |
| Saturation boost | 11 | 11 | grid |
| Bilateral filter | 847 | 30 | TPE |
| CLAHE | 36 | 29 | TPE |
| Unsharp mask | 176 | 29 | TPE |
| Laplacian sharpening | 32 | 29 | TPE |

**Evaluation (`src/yolo.py`)**
- The confidence threshold used to compute mAP50 is configurable; the default
  is 0.001 (standard value) instead of 0.25.
- GPU memory is freed after every evaluation (`gc.collect()` and
  `torch.cuda.empty_cache()`). Without this, memory grew at each trial and the
  search crashed with an out-of-memory error after about 15 trials.

**Scripts**
- `train_yolo.py` trains YOLOv3u on the HCM images, with or without the
  targeted data augmentation, and evaluates the best model on the HCM and LCM
  test sets. It replaces `yolo_training.ipynb`, using explicit train/val/test
  splits and applying the augmentation filters in BGR order, as in the search.
- `run_filter_search.py` runs the filter search for a trained model and
  evaluates the resulting pipeline on the LCM test set. It replaces
  `training.ipynb`.

## Installation

Tested with Python 3.9.21, PyTorch 2.5.1 (CUDA 12.1), Ultralytics 8.4.147,
Optuna 5.0.0, OpenCV 5.0.0 and NumPy 2.0.2, on NVIDIA A30 and L40S GPUs.

```bash
git clone https://github.com/davidevaler/filter_learning_framework.git
cd filter_learning_framework
python -m venv venv
source venv/bin/activate
pip install -e .
# optional, to browse the Optuna studies
pip install -e ".[dashboard]"
```

## Dataset

The experiments use the public 1000x subset of the M5 dataset prepared by the
original authors
([Kaggle: davidesenette/malaria-hcm-lcm-1000](https://www.kaggle.com/datasets/davidesenette/malaria-hcm-lcm-1000)):
837 training, 45 validation and 375 test images for each microscope. Place it
in `resources/dataset/` with this structure:

```
resources/dataset/
├── source/                 # HCM
│   ├── images/{train,val,test}/*.png
│   └── labels/{train,val,test}/*.txt
└── target/                 # LCM
    ├── images/{train,val,test}/*.png
    └── labels/{train,val,test}/*.txt
```

Class IDs: 0 schizont, 1 gametocyte, 2 ring, 3 trophozoite.

## Usage

All commands are run from the root of the repository.

**1. Train a model on HCM images**

```bash
# baseline, without data augmentation
python train_yolo.py --dataset resources/dataset --name no_da_s0 --seed 0
# with the targeted data augmentation
python train_yolo.py --dataset resources/dataset --name with_da_s0 --seed 0 --da
```

The model is saved in `runs/<name>/weights/best.pt` (best epoch on HCM
validation), and the test results in `runs/<name>/summary.json`.

**2. Search the filter pipeline for a trained model**

```bash
# on the 45 LCM validation images
python run_filter_search.py --model runs/with_da_s0/weights/best.pt --name with_da_s0
# on 200 random LCM training images
python run_filter_search.py --model runs/with_da_s0/weights/best.pt \
    --name with_da_s0_train200 --search-split train --search-n 200
```

The results are saved in `runs/search/<name>/`:

| File | Content |
|------|---------|
| `result.json` | mAP50 with and without the pipeline, trials used, search time |
| `pipeline.json` | the pipeline found, as a list of filters and parameters |
| `checkpoint.json` | full log of the search |
| `optuna.db` | Optuna studies (`optuna-dashboard sqlite:///runs/search/<name>/optuna.db`) |

`pipeline.json` and `checkpoint.json` can be loaded directly by the
*Malaria Preprocessing* KNIME node.

Both scripts need a GPU: training takes about 60–90 minutes per model, a
search on the 45 validation images about 50 minutes, and a search on 200
images 4–7.5 hours (NVIDIA A30 / L40S).

## Results

mAP50 on the LCM test set, mean ± standard deviation over three trainings
(seeds 0, 1, 2), models trained on HCM images only:

| Configuration | HCM test | LCM test |
|---------------|---------:|---------:|
| Baseline (no data augmentation, no filters) | 0.649 ± 0.015 | 0.188 ± 0.034 |
| Filters only | – | 0.183 ± 0.050 |
| Data augmentation only | 0.619 ± 0.032 | 0.267 ± 0.050 |
| Data augmentation + filters | – | 0.278 ± 0.046 |

Almost all the improvement comes from the targeted data augmentation. The
details are in the thesis.

## Credits

Original framework, dataset preparation and targeted data augmentation:
Davide Senette and Davide Andreolli. Changes listed above: Davide Valer.