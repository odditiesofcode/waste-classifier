"""
Tests for src/data.py.

Why test data-splitting code specifically: a bug here is dangerous in a
quiet way. Code that trains a model will usually crash loudly if it's
broken. Code that silently leaks test images into the training set, or
skews the split, produces a model that looks great on paper and fails
in the real world — with no error message telling you why. These tests
exist to catch exactly that class of bug, and to keep catching it if
anyone (including future-us) edits this logic later.
"""

import pandas as pd
import pytest

from src.data import stratified_split, compute_class_weights, build_manifest, DATA_DIR


@pytest.fixture
def fake_manifest():
    """A small synthetic manifest so tests run instantly and don't
    depend on the real dataset being present (e.g. in CI, where we may
    not want to fetch 43MB of images just to run unit tests)."""
    rows = []
    # deliberately imbalanced, like the real dataset
    class_counts = {"a": 100, "b": 60, "c": 10}
    for cls, n in class_counts.items():
        for i in range(n):
            rows.append({"path": f"/fake/{cls}/{i}.jpg", "class": cls})
    return pd.DataFrame(rows)


def test_split_sizes_are_close_to_requested_fractions(fake_manifest):
    result = stratified_split(fake_manifest, train_frac=0.7, val_frac=0.15, test_frac=0.15)
    n = len(fake_manifest)
    counts = result["split"].value_counts()
    assert abs(counts["train"] / n - 0.7) < 0.02
    assert abs(counts["val"] / n - 0.15) < 0.02
    assert abs(counts["test"] / n - 0.15) < 0.02


def test_no_image_appears_in_more_than_one_split(fake_manifest):
    """The single most important property of any split: zero leakage."""
    result = stratified_split(fake_manifest)
    seen_paths = result["path"]
    assert seen_paths.is_unique, "an image path appears more than once across splits"


def test_every_image_is_assigned_to_a_split(fake_manifest):
    result = stratified_split(fake_manifest)
    assert len(result) == len(fake_manifest)
    assert result["split"].isin(["train", "val", "test"]).all()


def test_class_proportions_are_preserved_across_splits(fake_manifest):
    """Stratification should mean each split has roughly the same class
    mix as the full dataset — this is what makes eval numbers trustworthy."""
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
    """The smallest class ('c', 10 samples) should get the largest
    weight, since that's the entire point of computing these."""
    split_df = stratified_split(fake_manifest)
    weights = compute_class_weights(split_df)

    assert weights["c"] > weights["b"] > weights["a"]


def test_class_weights_computed_only_from_train_split(fake_manifest):
    """Regression test for a real leakage risk: weights must not change
    if we corrupt the val/test portions of the class distribution."""
    split_df = stratified_split(fake_manifest)
    weights_before = compute_class_weights(split_df)

    # corrupt every non-train row's class label
    corrupted = split_df.copy()
    corrupted.loc[corrupted["split"] != "train", "class"] = "a"
    weights_after = compute_class_weights(corrupted)

    assert weights_before == weights_after


@pytest.mark.skipif(not DATA_DIR.exists(), reason="real dataset not present in this environment")
def test_real_dataset_manifest_matches_known_counts():
    """Sanity check against the actual TrashNet dataset, when available
    (e.g. when running locally, not necessarily in CI)."""
    manifest = build_manifest()
    counts = manifest["class"].value_counts().to_dict()
    assert counts == {
        "paper": 594,
        "glass": 501,
        "plastic": 482,
        "metal": 410,
        "cardboard": 403,
        "trash": 137,
    }
