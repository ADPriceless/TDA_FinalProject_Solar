"""Tests for creating annotation files for model training"""


# pylint: disable=missing-function-docstring


import csv
from pathlib import Path

import pytest

from utils.make_annotation_file import (
    write_img_and_mask_names_to_csv,
    write_annotation_files
)


# ---------------------------------------------------------------------------
# Global variables
DATA_DIR = Path('data/Hou/PV01_Rooftop_Brick')
CSV_PATH = Path('test/annotation.csv')
DIRS_UNDER_TEST = list(Path('data/Hou').iterdir())


# ---------------------------------------------------------------------------
# Tests
def test_makes_csv():
    write_img_and_mask_names_to_csv(DATA_DIR, CSV_PATH)
    assert CSV_PATH.exists()


def test_populates_csv():
    write_img_and_mask_names_to_csv(DATA_DIR, CSV_PATH)
    lines = read_csv(CSV_PATH)
    assert len(lines[0]) == 2


def test_filenames_only():
    """Assert that each filepath is only the filename"""
    write_img_and_mask_names_to_csv(DATA_DIR, CSV_PATH)
    for line in read_csv(CSV_PATH):
        assert Path(line[0]).name == line[0]
        assert Path(line[1]).name == line[1]


def test_annotation_files_written():
    write_annotation_files(DIRS_UNDER_TEST)
    for directory in DIRS_UNDER_TEST:
        assert directory.joinpath('annotation.csv').exists()


def test_annotation_files_written_to_correct_dirs():
    write_annotation_files(DIRS_UNDER_TEST)
    for directory in DIRS_UNDER_TEST:
        assert directory.joinpath('annotation.csv').exists()


def test_annotate_files_not_overwritten():
    # create dummy files
    for directory in DIRS_UNDER_TEST:
        with open(
            directory.joinpath('annotation.csv'), 'w', encoding='utf-8'
        ):
            pass
    # run test
    write_annotation_files(DIRS_UNDER_TEST)
    # assert that dummy files are empty
    for directory in DIRS_UNDER_TEST:
        lines = read_csv(directory.joinpath('annotation.csv'))
        assert len(lines) == 0


def test_label_always_second():
    write_annotation_files(DIRS_UNDER_TEST)
    for directory in DIRS_UNDER_TEST:
        lines = read_csv(directory.joinpath('annotation.csv'))
        # assert that only the second filename ends with "label"
        for line in lines:
            assert not line[0].split('.')[0].endswith('label')
            assert line[1].split('.')[0].endswith('label')


# ---------------------------------------------------------------------------
# Fixtures
@pytest.fixture(scope='module', autouse=True)
def cleanup():
    yield
    _cleanup_csv()
    _cleanup_dirs_ut()


def _cleanup_csv():
    if CSV_PATH.exists():
        CSV_PATH.unlink()


def _cleanup_dirs_ut():
    for dir_ in DIRS_UNDER_TEST:
        if dir_.joinpath('annotation.csv').exists():
            dir_.joinpath('annotation.csv').unlink()


# --------------------------------------------------------------------------
# Helper functions
def read_csv(csv_path: Path) -> list:
    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        lines = list(reader)
    return lines
