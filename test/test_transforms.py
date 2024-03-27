"""Test transformations on images from datsets."""

# pylint: disable=missing-function-docstring

from pathlib import Path

import pytest
from torch.utils.data import DataLoader
import torchvision.transforms as tvt

from preprocess.datasets import make_kasmi_ign_dataset
from preprocess.loaders import make_hou_loader
from preprocess.transforms import scale_pixels, binarise_mask, one_hot, rgba_to_rgb


HOU_CROPLAND_DIR = Path('data/Hou/PV03_Ground_Cropland')


def test_resize_image():
    hou_ds = make_hou_loader(
        HOU_CROPLAND_DIR,
        img_transform=tvt.Resize((256, 256)),
        mask_transform=tvt.Resize((126, 126))
    )
    for img, mask in hou_ds:
        assert img.shape == (1, 3, 256, 256)
        assert mask.shape == (1, 1, 126, 126)


@pytest.mark.skip(reason='Function not used')
def test_scale_img_pixel_values():
    hou_ds = make_hou_loader(
        HOU_CROPLAND_DIR,
        img_transform=tvt.Lambda(scale_pixels),
    )
    count = 0
    for img, _ in hou_ds:
        assert img.shape == (1, 3, 1024, 1024)
        assert img.max() <= 1
        assert img.type() == 'torch.FloatTensor'
        if 0 < img[0, 0, 0, 0] < 1:
            count += 1
    assert count > 0 # check that there are values between 0 and 1


@pytest.mark.skip(reason='Function obsolete due to `one_hot`')
def test_binarise_mask():
    hou_ds = make_hou_loader(
        HOU_CROPLAND_DIR,
        mask_transform=tvt.Lambda(binarise_mask),
    )
    count = 0
    for _, mask in hou_ds:
        n_unique = len(mask.unique())
        if n_unique == 2:
            count += 1
        assert n_unique in (1, 2)
        # Pixel values in `mask` where there are solar
        # panels must be one.
        if n_unique == 2:
            assert mask.max() == 1
        # Pixel values in `mask` where there are no solar
        # panels must be zero.
        assert mask.min() == 0
    assert count > 0 # check that there are outputs with two unique values


def test_composed_transforms():
    img_transform = tvt.Compose([tvt.Resize((256, 256)), tvt.Lambda(scale_pixels)])
    mask_transform = tvt.Compose([tvt.Resize((256, 256)), tvt.Lambda(binarise_mask)])
    hou_ds = make_hou_loader(
        HOU_CROPLAND_DIR,
        img_transform=img_transform,
        mask_transform=mask_transform
    )
    for img, mask in hou_ds:
        assert img.shape == (1, 3, 256, 256)
        assert img.max() <= 1

        assert mask.shape == (1, 1, 256, 256)
        n_unique = len(mask.unique())
        # if n_unique == 2:
        #     count += 1
        assert n_unique in (1, 2)
         # Pixel values in `mask` where there are solar
        # panels must be one.
        if n_unique == 2:
            assert mask.max() == 1
        # Pixel values in `mask` where there are no solar
        # panels must be zero.
        assert mask.min() == 0


def test_one_hot():
    mask_size = 128
    params = {
        'batch_size': 32,
        'drop_last': True,
    }
    mask_transform = tvt.Compose([
        tvt.Resize((mask_size, mask_size)),
        tvt.Lambda(one_hot),
    ])
    hou_ds = make_hou_loader(
        HOU_CROPLAND_DIR,
        mask_transform=mask_transform,
        params=params
    )
    for _, mask in hou_ds:
        assert mask.shape == (params['batch_size'], 2, mask_size, mask_size)
        # Assert that mask class 0 is the opposite of mask class 1
        # so the sum of the mask is equal to the number of pixels
        # multiplied by the batch size.
        assert mask.sum() == params['batch_size'] * mask_size * mask_size


def test_rgba_to_rgb():
    transform = tvt.Lambda(rgba_to_rgb)
    ign_ds = make_kasmi_ign_dataset(img_transforms=transform)
    loader = DataLoader(ign_ds)
    for img, _ in loader:
        assert img.shape == (1, 3, 400, 400)
