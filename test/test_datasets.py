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

from preprocess.datasets import HouDataset
from utils.make_annotation_file import write_annotation_files


# ---------------------------------------------------------------------------
# Global variables
DIRS_UNDER_TEST = list(Path('data/Hou').iterdir())
DIR_LENGTHS = (138, 413, 94, 859, 117, 352, 119, 625, 236)
DS_SUBSET_LENGTHS = \
    dict(zip([dir_.parts[-1] for dir_ in DIRS_UNDER_TEST], DIR_LENGTHS))


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


# ---------------------------------------------------------------------------
# Fixtures
@pytest.fixture
def hou_ds_brick():
    """Useful for an initial dataset to test due to its
    small size."""
    ds_dir = Path('data/Hou/PV01_Rooftop_Brick')
    annotation_file = ds_dir.joinpath('annotation.csv')
    write_annotation_files([ds_dir])
    return HouDataset(annotation_file, ds_dir)


@pytest.fixture
def hou_ds_rand_subset():
    """Useful to gain more coverage of the whole dataset
    over multiple test runs, without taking too long."""
    ds_dir = np.random.choice(DIRS_UNDER_TEST)
    annotation_file = ds_dir.joinpath('annotation.csv')
    write_annotation_files([ds_dir])
    return HouDataset(annotation_file, ds_dir)


@pytest.fixture(scope='module', autouse=True)
def cleanup():
    yield
    _cleanup_dirs_ut()


def _cleanup_dirs_ut():
    for dir_ in DIRS_UNDER_TEST:
        if dir_.joinpath('annotation.csv').exists():
            dir_.joinpath('annotation.csv').unlink()
