"""
Tests for src/data.py.

Why test data-splitting code specifically: a bug here is dangerous in a
quiet way. Code that trains a model will usually crash loudly if it's
broken. Code that silently leaks test images into the training set, or
skews the split, produces a model that looks great on paper and fails
in the real world -- with no error message telling you why.
"""

import pandas as pd
import pytest

from src.data import (
    stratified_split,
    compute_class_weights,
    build_manifest,
    DATA_DIR,
    REALWASTE_CLASS_MAP,
    reserve_real_world_holdout,
)


@pytest.fixture
def fake_manifest():
    """A small synthetic manifest so tests run instantly and don't
    depend on the real dataset being present."""
    rows = []
    class_counts = {"a": 100, "b": 60, "c": 10}
    for cls, n in class_counts.items():
        for i in range(n):
            rows.append({"path": f"/fake/{cls}/{i}.jpg", "class": cls})
    return pd.DataFrame(rows)


@pytest.fixture
def fake_realwaste_manifest():
    """Synthetic RealWaste-shaped manifest (already class-mapped),
    imbalanced like the real one, with a 'source' column."""
    rows = []
    class_counts = {"cardboard": 40, "glass": 30, "metal": 50, "trash": 80}
    for cls, n in class_counts.items():
        for i in range(n):
            rows.append({"path": f"/fake_rw/{cls}/{i}.jpg", "class": cls, "source": "realwaste"})
    return pd.DataFrame(rows)


# --- existing split tests (unchanged behavior, still must hold) -----------

def test_split_sizes_are_close_to_requested_fractions(fake_manifest):
    result = stratified_split(fake_manifest, train_frac=0.7, val_frac=0.15, test_frac=0.15)
    n = len(fake_manifest)
    counts = result["split"].value_counts()
    assert abs(counts["train"] / n - 0.7) < 0.02
    assert abs(counts["val"] / n - 0.15) < 0.02
    assert abs(counts["test"] / n - 0.15) < 0.02


def test_no_image_appears_in_more_than_one_split(fake_manifest):
    result = stratified_split(fake_manifest)
    assert result["path"].is_unique, "an image path appears more than once across splits"


def test_every_image_is_assigned_to_a_split(fake_manifest):
    result = stratified_split(fake_manifest)
    assert len(result) == len(fake_manifest)
    assert result["split"].isin(["train", "val", "test"]).all()


def test_class_proportions_are_preserved_across_splits(fake_manifest):
    result = stratified_split(fake_manifest)
    full_props = fake_manifest["class"].value_counts(normalize=True)
    for split_name in ["train", "val", "test"]:
        split_props = result[result["split"] == split_name]["class"].value_counts(normalize=True)
        for cls in full_props.index:
            assert abs(split_props.get(cls, 0) - full_props[cls]) < 0.05, (
                f"class '{cls}' proportion drifted too much in split '{split_name}'"
            )


def test_split_fractions_must_sum_to_one(fake_manifest):
    with pytest.raises(AssertionError):
        stratified_split(fake_manifest, train_frac=0.5, val_frac=0.5, test_frac=0.5)


def test_class_weights_favor_minority_classes(fake_manifest):
    split_df = stratified_split(fake_manifest)
    weights = compute_class_weights(split_df)
    assert weights["c"] > weights["b"] > weights["a"]


def test_class_weights_computed_only_from_train_split(fake_manifest):
    split_df = stratified_split(fake_manifest)
    weights_before = compute_class_weights(split_df)
    corrupted = split_df.copy()
    corrupted.loc[corrupted["split"] != "train", "class"] = "a"
    weights_after = compute_class_weights(corrupted)
    assert weights_before == weights_after


@pytest.mark.skipif(not DATA_DIR.exists(), reason="real dataset not present in this environment")
def test_real_dataset_manifest_matches_known_counts():
    manifest = build_manifest()
    counts = manifest["class"].value_counts().to_dict()
    assert counts == {
        "paper": 594, "glass": 501, "plastic": 482,
        "metal": 410, "cardboard": 403, "trash": 137,
    }


# --- new: RealWaste integration tests --------------------------------------

def test_realwaste_class_map_covers_all_nine_official_classes():
    """Regression test: if UCI ever adds/renames a class, or we typo a
    key, we want a loud test failure -- not images silently getting
    dropped during build_realwaste_manifest()."""
    expected = {
        "Cardboard", "Glass", "Metal", "Paper", "Plastic",
        "Food Organics", "Miscellaneous Trash", "Textile Trash", "Vegetation",
    }
    assert set(REALWASTE_CLASS_MAP.keys()) == expected


def test_realwaste_class_map_only_produces_our_six_classes():
    our_classes = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
    assert set(REALWASTE_CLASS_MAP.values()) == our_classes


def test_holdout_and_trainable_do_not_overlap(fake_realwaste_manifest):
    """The single most important property: zero leakage between the
    real-world holdout and anything used in training."""
    holdout_df, trainable_df = reserve_real_world_holdout(fake_realwaste_manifest)
    holdout_paths = set(holdout_df["path"])
    trainable_paths = set(trainable_df["path"])
    assert holdout_paths.isdisjoint(trainable_paths)


def test_holdout_covers_every_image_exactly_once(fake_realwaste_manifest):
    holdout_df, trainable_df = reserve_real_world_holdout(fake_realwaste_manifest)
    assert len(holdout_df) + len(trainable_df) == len(fake_realwaste_manifest)


def test_holdout_fraction_is_approximately_correct(fake_realwaste_manifest):
    holdout_df, trainable_df = reserve_real_world_holdout(fake_realwaste_manifest, holdout_frac=0.3)
    n = len(fake_realwaste_manifest)
    assert abs(len(holdout_df) / n - 0.3) < 0.03


def test_holdout_is_stratified_by_class(fake_realwaste_manifest):
    """A holdout skewed toward one class (purely by chance) would make
    our generalization-gap measurement noisy and untrustworthy."""
    holdout_df, _ = reserve_real_world_holdout(fake_realwaste_manifest)
    full_props = fake_realwaste_manifest["class"].value_counts(normalize=True)
    holdout_props = holdout_df["class"].value_counts(normalize=True)
    for cls in full_props.index:
        assert abs(holdout_props.get(cls, 0) - full_props[cls]) < 0.05


def test_holdout_split_is_reproducible(fake_realwaste_manifest):
    """Same seed must give the same holdout every time -- otherwise
    'never used in training' silently stops being true across reruns."""
    holdout_1, _ = reserve_real_world_holdout(fake_realwaste_manifest, seed=42)
    holdout_2, _ = reserve_real_world_holdout(fake_realwaste_manifest, seed=42)
    assert set(holdout_1["path"]) == set(holdout_2["path"])
