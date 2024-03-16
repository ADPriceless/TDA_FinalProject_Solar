"""Tests for creating annotation files for model training"""


# pylint: disable=missing-function-docstring


import csv
from pathlib import Path

import pytest

from global_variables import (
    ANNOTATION_PATH, DIRS_UNDER_TEST, HOU_BRICK_DIR, ROOT_HOU
)
from utils.make_annotation_file import (
    get_img_and_mask_filepaths,
    path_excluded,
    write_path_of_img_and_mask_to_csv,
    create_annotation_files_hou,
)


# ---------------------------------------------------------------------------
# Tests
def test_get_img_and_mask_filepaths_returns_pair():
    gen = HOU_BRICK_DIR.glob('**/*.png')
    assert len(get_img_and_mask_filepaths(gen)) == 2


def test_get_img_and_mask_filepaths_label_always_second():
    gen = HOU_BRICK_DIR.glob('**/*.png')
    # using `for` instead of `while` to avoid risk of infinite loop
    for _ in range(300):
        try:
            img, mask = get_img_and_mask_filepaths(gen)
            assert not img.as_posix().endswith('label.png')
            assert mask.as_posix().endswith('label.png')
        except StopIteration:
            break
    else:
        assert False # if here, code did not break out of `for` loop


def test_path_excluded():
    split_idx = 3
    exclude_dirs = DIRS_UNDER_TEST[:split_idx]
    # Check first i directories are excluded
    for i in range(split_idx):
        for file in DIRS_UNDER_TEST[i].iterdir():
            assert path_excluded(file, exclude_dirs)
    # Check remaining directories are not excluded
    for i in range(split_idx, len(DIRS_UNDER_TEST)):
        for file in DIRS_UNDER_TEST[i].iterdir():
            assert not path_excluded(file, exclude_dirs)


def test_write_path_of_img_and_mask_to_csv():
    gen = HOU_BRICK_DIR.glob('**/*.png')
    # using `for` instead of `while` to avoid risk of infinite loop
    for _ in range(300):
        try:
            img, mask = get_img_and_mask_filepaths(gen)
            write_path_of_img_and_mask_to_csv(img, mask, ANNOTATION_PATH)
        except StopIteration:
            break
    else:
        assert False # if here, code did not break out of `for` loop
    for line in read_csv(ANNOTATION_PATH):
        _annotation_general_checks(line)


def test_create_annotation_files_hou_does_not_overwrite_existing():
    with open(ANNOTATION_PATH, 'w', encoding='utf-8'):
        pass
    create_annotation_files_hou(ANNOTATION_PATH, HOU_BRICK_DIR)
    assert len(read_csv(ANNOTATION_PATH)) == 0


def test_create_annotation_files_hou_ds_subset():
    create_annotation_files_hou(ANNOTATION_PATH, HOU_BRICK_DIR)
    for line in read_csv(ANNOTATION_PATH):
        _annotation_general_checks(line)
        # Check path is relative. In this case, there should
        # only be the filename.
        assert len(Path(line[0]).parts) == 1
        assert len(Path(line[1]).parts) == 1


def test_create_annotation_files_hou_whole_ds():
    create_annotation_files_hou(ANNOTATION_PATH, ROOT_HOU)
    for line in read_csv(ANNOTATION_PATH):
        _annotation_general_checks(line)
        # Check path is relative.
        assert len(Path(line[0]).parts) > 1
        assert len(Path(line[1]).parts) > 1


# ---------------------------------------------------------------------------
# Fixtures
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


# --------------------------------------------------------------------------
# Helper functions
def read_csv(csv_path: Path) -> list[list[str]]:
    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        lines = list(reader)
    return lines


def _annotation_general_checks(line: list[str]) -> None:
    assert len(line) == 2
    img_path, mask_path = line
    assert not img_path.endswith('label.png')
    assert mask_path.endswith('label.png')
