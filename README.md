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
- [x] Evaluation report (confusion matrix, per-class metrics, error analysis)
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
python -m src.train --epochs 15 --batch-size 32
```

Best checkpoint (by validation accuracy) is saved to
`checkpoints/best_model.pt`, alongside `checkpoints/training_history.json`.

Trained run: **88.4% validation accuracy** after 15 epochs.

## Evaluation

```bash
python -m src.evaluate
```

Runs the best checkpoint against the held-out **test** set (never used
for any training decision, including checkpoint selection — that used
val, so test is our least-biased read on real performance). Produces a
per-class precision/recall/F1 report, a confusion matrix image, and a
CSV of every misclassified image sorted by prediction confidence.

### Results

**Overall test accuracy: 91.6%** (380 test images, 32 misclassified)

| Class     | Precision | Recall | F1    | Support |
|-----------|-----------|--------|-------|---------|
| cardboard | 0.965     | 0.917  | 0.940 | 60      |
| glass     | 0.957     | 0.868  | 0.910 | 76      |
| metal     | 0.792     | 0.984  | 0.878 | 62      |
| paper     | 0.945     | 0.966  | 0.956 | 89      |
| plastic   | 0.926     | 0.863  | 0.894 | 73      |
| trash     | 0.944     | 0.850  | 0.895 | 20      |

### Error analysis

**Hypothesis going in (from EDA):** `trash` — the smallest class (137
training images) and the most visually inconsistent (a grab-bag of
unrelated items rather than one visual concept) — would be the weakest
performer, imbalance handling notwithstanding.

**What actually happened:** `trash` performed well (0.944 precision,
0.850 recall), on par with the strongest classes. This is a good signal
that the inverse-frequency class weighting in `src/data.py` (`trash`
got a 3.07x loss weight vs. `paper`'s 0.71x) did what it was meant to —
it isn't just a theoretical fix, it visibly changed the outcome.

**The real weak point turned out to be `metal`** — and its
precision/recall gap (0.792 vs. 0.984) tells a specific story: the
model rarely *misses* actual metal, but it over-*predicts* metal on
things that aren't. Metal has effectively become a fallback guess for
uncertain images.

Inspecting the four most confidently-wrong test predictions (all
misclassified as `metal`, at 90-99.9% confidence) showed:

- A glass bottle photographed upside-down, cropped tight on a gold
  foil cap and metallic-looking rim.
- A matte-black plastic protein jar with a glossy black cap — reads
  visually as an industrial metal canister rather than typical plastic
  packaging.
- A glass jar's threaded neck/rim in extreme close-up — genuinely
  looks like a metal lid even to a human eye at that crop.
- One cardboard scrap with no metallic features at all — this one
  doesn't fit the pattern above, so the "metallic caps/rims" story is
  a partial explanation, not a complete one.

**Working theory:** the model may be leaning on caps, rims, and
close-up neck crops as a proxy signal for "metal," which is a
reasonable thing to learn given that actual metal cans in this dataset
are essentially all rim/neck-shaped objects — but it misfires on other
classes' caps and close crops.

**Possible next steps** (not yet implemented):
- Reduce how aggressively `RandomResizedCrop` zooms during training,
  so the model sees fewer isolated cap/rim crops without surrounding
  context.
- Add more varied crops/angles of glass and plastic containers,
  specifically ones featuring caps and rims, to break the spurious
  correlation.
- Try focal loss instead of fixed class weights, since this failure
  mode isn't about class frequency at all — plain reweighting can't
  fix it.

## Project structure

```
data/            raw dataset (gitignored) + splits.csv (tracked)
notebooks/       EDA
src/             data splitting, dataset/model classes, training, evaluation
tests/           unit tests (run in CI)
scripts/         reproducible data download
checkpoints/     trained weights + evaluation outputs (gitignored)
```
