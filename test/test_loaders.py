"""Tests for the loader module."""


# pylint: disable=missing-docstring


from pathlib import Path

import pytest
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from global_variables import ROOT_HOU, HOU_BRICK_DIR, DS_SUBSET_LENGTHS
from preprocess.loaders import make_hou_loader, make_kasmi_loader


def test_returns_dataloader():
    loader = make_hou_loader(HOU_BRICK_DIR)
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 1
    assert isinstance(loader.sampler, SequentialSampler)
    assert len(loader.dataset) == DS_SUBSET_LENGTHS[HOU_BRICK_DIR.name]


def test_sample_loading():
    for sample in make_hou_loader(HOU_BRICK_DIR):
        assert len(sample) == 2
        img, mask = sample
        assert img.shape == (1, 3, 256, 256)
        assert mask.shape == (1, 1, 256, 256)


def test_sampler():
    loader = make_hou_loader(HOU_BRICK_DIR)
    assert isinstance(loader.sampler, SequentialSampler)
    # Sampler returns the indices of samples in the same
    # order, unless `shuffle=True` is specified.
    for count, index in enumerate(loader.sampler):
        assert index == count


def test_batch_sampler():
    params = {
        'batch_size': 32,
        'drop_last': True,
    }
    loader = make_hou_loader(HOU_BRICK_DIR, params=params)
    assert loader.batch_size == params['batch_size']
    for indices in loader.batch_sampler:
        assert len(indices) == params['batch_size']
    params['batch_size'] = 64
    loader = make_hou_loader(HOU_BRICK_DIR, params=params)
    assert loader.batch_size == params['batch_size']
    for indices in loader.batch_sampler:
        assert len(indices) == params['batch_size']


def test_batch_sample_loading():
    params = {
        'batch_size': 32,
        'drop_last': True,
    }
    loader = make_hou_loader(HOU_BRICK_DIR, params=params)
    for img_batch, mask_batch in loader:
        assert img_batch.shape == (params['batch_size'], 3, 256, 256)
        assert mask_batch.shape == (params['batch_size'], 1, 256, 256)
    # test shuffling
    params['shuffle'] = True
    loader_shuffle = make_hou_loader(HOU_BRICK_DIR, params=params)
    for img_batch, mask_batch in loader_shuffle:
        assert img_batch.shape == (params['batch_size'], 3, 256, 256)
        assert mask_batch.shape == (params['batch_size'], 1, 256, 256)


def test_bad_params():
    params = {
        'batch_size': 32,
        'shuffle': True,
        'nonsense': 'foobarbuzz'
    }
    with pytest.raises(TypeError):
        make_hou_loader(HOU_BRICK_DIR, params=params)


def test_wrong_arg_order():
    params = {
        'batch_size': 32
    }
    with pytest.raises(TypeError):
        make_hou_loader(ROOT_HOU, params)


def test_bad_types():
    bad_path_type = 'some/path/as/str'
    bad_param_type = (32,)
    with pytest.raises(TypeError):
        make_hou_loader(bad_path_type)
    with pytest.raises(TypeError):
        make_hou_loader(HOU_BRICK_DIR, [bad_path_type])
    with pytest.raises(TypeError):
        make_hou_loader(HOU_BRICK_DIR, bad_path_type)
    with pytest.raises(TypeError):
        make_hou_loader(HOU_BRICK_DIR, 100)
    with pytest.raises(TypeError):
        make_hou_loader(HOU_BRICK_DIR, params=bad_param_type)


def test_make_kasmi_loader_not_implemented():
    """This should be implemented later, but leaving
    a placeholder for now."""
    with pytest.raises(NotImplementedError):
        make_kasmi_loader(Path('some/path/'))
