"""Tests for creating annotation files for model training"""


# pylint: disable=missing-function-docstring


import csv
from pathlib import Path

import pytest

from utils.make_annotation_file import write_img_and_mask_names_to_csv


# ---------------------------------------------------------------------------
# Global variables
DATA_DIR = Path('data/Hou/PV01_Rooftop_Brick')
CSV_PATH = Path('test/annotation.csv')


# ---------------------------------------------------------------------------
# Tests
def test_makes_csv():
    write_img_and_mask_names_to_csv(DATA_DIR, CSV_PATH)
    assert CSV_PATH.exists()


def test_populates_csv():
    write_img_and_mask_names_to_csv(DATA_DIR, CSV_PATH)
    lines = read_csv()
    assert len(lines[0]) == 2


def test_filenames_only():
    """Assert that each filepath is only the filename"""
    write_img_and_mask_names_to_csv(DATA_DIR, CSV_PATH)
    for line in read_csv():
        assert Path(line[0]).name == line[0]
        assert Path(line[1]).name == line[1]


# ---------------------------------------------------------------------------
# Fixtures
@pytest.fixture(scope='module', autouse=True)
def cleanup():
    yield
    _cleanup_csv()

def _cleanup_csv():
    if CSV_PATH.exists():
        CSV_PATH.unlink()


# --------------------------------------------------------------------------
# Helper functions
def read_csv() -> list:
    with open(CSV_PATH, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        lines = list(reader)
    return lines
