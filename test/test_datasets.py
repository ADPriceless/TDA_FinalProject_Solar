"""Tests to validate class representations of datasets"""


# pylint: disable=missing-docstring,consider-using-enumerate,redefined-outer-name

# pylint `consider-using enumerate` appears for the lines `for
# sample in range(len(hou_ds)):`. It is disabled because using
# a statement like `for sample in hou_ds` exceeds the length
# of the dataset. I believe this is because the dataset has no
# iterator implementation, and currently I'm not planning to
# add this.


from pathlib import Path

import numpy as np
import pytest

from global_variables import (
    ANNOTATION_PATH, DIRS_UNDER_TEST, DS_SUBSET_LENGTHS, HOU_BRICK_DIR
)
from preprocess.datasets import ImgAndMaskDataset, make_kasmi_ign_dataset
from utils.make_annotation_file import create_annotation_files_hou


# ---------------------------------------------------------------------------
# Tests
def test_ds_len_correct(hou_ds_rand_subset):
    expected_len = DS_SUBSET_LENGTHS[hou_ds_rand_subset.img_dir.parts[-1]]
    assert len(hou_ds_rand_subset) == expected_len


def test_ds_loads_img_mask_pair(hou_ds_rand_subset):
    for sample in range(len(hou_ds_rand_subset)):
        assert len(hou_ds_rand_subset[sample]) == 2


def test_img_mask_shape(hou_ds_rand_subset):
    for sample in range(len(hou_ds_rand_subset)):
        img, mask = hou_ds_rand_subset[sample]
        # shape should be (n_channels, width, height)
        assert img.shape  in ((3, 256, 256), (3, 1024, 1024)) # RGB
        assert mask.shape in ((1, 256, 256), (1, 1024, 1024)) # 1D


def test_mask_is_binary(hou_ds_rand_subset):
    for sample in range(len(hou_ds_rand_subset)):
        _, mask = hou_ds_rand_subset[sample]
        # Only 1 unique value in `mask` if no solar panels in
        # `img`. Otherwise, there should be 2 unique values.
        n_unique = len(mask.unique())
        assert n_unique in (1, 2)
        # Any value from 1 to 255 is acceptable for
        # the masks' area
        if n_unique == 2:
            assert mask.max() > 0
        # Pixel values in `mask` where there are no solar
        # panels must be zero.
        assert mask.min() == 0


def test_pixel_value_range(hou_ds_rand_subset):
    for sample in range(len(hou_ds_rand_subset)):
        img, _ = hou_ds_rand_subset[sample]
        # Pixel values in `img` should be in the range [0, 255]
        assert img.min() >= 0
        assert img.max() <= 255


def test_ds_raises_if_img_dir_is_file():
    create_annotation_files_hou(ANNOTATION_PATH, HOU_BRICK_DIR)
    with pytest.raises(ValueError):
        ImgAndMaskDataset(ANNOTATION_PATH, ANNOTATION_PATH)


def test_ds_raises_if_annotation_file_not_csv():
    annotation_file = HOU_BRICK_DIR.joinpath('annotation.txt')
    _create_file(annotation_file)
    with pytest.raises(ValueError):
        ImgAndMaskDataset(HOU_BRICK_DIR, annotation_file)
    _cleanup_txt_files()


def test_make_kasmi_ign_dataset():
    ign_ds = make_kasmi_ign_dataset()
    assert isinstance(ign_ds, ImgAndMaskDataset)


def test_ign_ds_samples():
    blank_mask_shape = (1, 520, 520)
    ign_ds = make_kasmi_ign_dataset()
    for sample in range(len(ign_ds)):
        assert len(ign_ds[sample]) == 2
        img, mask = ign_ds[sample]
        assert img.shape in ((3, 400, 400), (4, 400, 400))
        assert img.max() <= 255
        assert img.min() >= 0
        assert mask.shape in ((1, 400, 400), blank_mask_shape)
        assert mask.max() <= 255
        assert mask.min() == 0
        assert len(mask.unique()) in (1, 2)



# ---------------------------------------------------------------------------
# Fixtures
@pytest.fixture
def hou_ds_brick():
    """Useful for an initial dataset to test due to its
    small size."""
    ds_dir = Path('data/Hou/PV01_Rooftop_Brick')
    create_annotation_files_hou(ANNOTATION_PATH, HOU_BRICK_DIR)
    return ImgAndMaskDataset(ANNOTATION_PATH, ds_dir)


@pytest.fixture
def hou_ds_rand_subset():
    """Useful to gain more coverage of the whole dataset
    over multiple test runs, without taking too long."""
    ds_dir = np.random.choice(DIRS_UNDER_TEST)
    create_annotation_files_hou(ANNOTATION_PATH, ds_dir)
    return ImgAndMaskDataset(ANNOTATION_PATH, ds_dir)


@pytest.fixture(autouse=True)
def cleanup():
    yield
    _cleanup_annotation()
    _cleanup_dirs_ut()


def _cleanup_annotation():
    if ANNOTATION_PATH.exists():
        ANNOTATION_PATH.unlink()


def _cleanup_dirs_ut():
    for dir_ in DIRS_UNDER_TEST:
        if dir_.joinpath('annotation.csv').exists():
            dir_.joinpath('annotation.csv').unlink()


def _create_file(path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('test')


def _cleanup_txt_files():
    for dir_ in DIRS_UNDER_TEST:
        if dir_.joinpath('annotation.txt').exists():
            dir_.joinpath('annotation.txt').unlink()
