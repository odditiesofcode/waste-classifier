# Waste Classifier

An end-to-end image classification project: given a photo of an item,
predict which of 6 waste/recycling categories it belongs to
(cardboard, glass, metal, paper, plastic, trash).

Built as a portfolio project focused on demonstrating the *full*
ML lifecycle — data validation, reproducible splits, class-imbalance
handling, transfer learning, evaluation, serving, and CI/CD — not just
a model in a notebook.

## Status

- [x] Data acquisition + EDA (`notebooks/eda.py`)
- [x] Reproducible, tested train/val/test split with class-weight
      computation (`src/data.py`, `tests/test_data.py`)
- [x] Training pipeline: transfer learning (ResNet18), augmentation,
      class-weighted loss, checkpointing (`src/train.py`)
- [ ] Evaluation report (confusion matrix, per-class metrics, error analysis)
- [ ] Real-world (non-studio-photo) holdout set for a true generalization check
- [ ] FastAPI serving layer
- [ ] Dockerfile
- [ ] GitHub Actions CI/CD

## Dataset

[TrashNet](https://github.com/garythung/trashnet) — 2,527 images across
6 classes, collected by Stanford students. Notably: clean studio photos
on plain backgrounds, and a real 4.3x class imbalance (594 `paper` vs.
137 `trash`). See `notebooks/eda.py` output for the full breakdown.

**Known limitation:** because the training data is entirely clean,
single-item studio photos, this model is expected to perform worse
on cluttered, real-world photos (e.g. actual bin/recycling photos).
We're building a small held-out set of real-world photos to measure
that gap explicitly rather than just assuming it away.

## Setup

```bash
pip install -r requirements.txt -r requirements-train.txt
bash scripts/download_data.sh
python src/data.py          # builds data/splits.csv
python -m pytest tests/     # sanity-check the data pipeline
```

## Training

Needs a GPU to be practical (CPU works but is slow). On
[Google Colab](https://colab.research.google.com): Runtime > Change
runtime type > T4 GPU, then run the setup steps above followed by:

```bash
python src/train.py --epochs 15 --batch-size 32
```

Best checkpoint (by validation accuracy) is saved to
`checkpoints/best_model.pt`, alongside `checkpoints/training_history.json`.

## Project structure

```
data/            raw dataset (gitignored) + splits.csv (tracked)
notebooks/       EDA
src/             data splitting, dataset/model classes, training
tests/           unit tests (run in CI)
scripts/         reproducible data download
checkpoints/     trained weights (gitignored)
```
