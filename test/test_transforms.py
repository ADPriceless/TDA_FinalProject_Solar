"""Test transformations on images from datsets."""

# pylint: disable=missing-function-docstring

from pathlib import Path

from torch.utils.data import DataLoader
import torchvision.transforms.v2 as tvt

from preprocess.datasets import make_kasmi_ign_dataset
from preprocess.loaders import make_hou_loader
from preprocess.transforms import one_hot, rgba_to_rgb


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
        assert img.type() == 'torch.ByteTensor'
        assert mask.type() == 'torch.ByteTensor'


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
        assert mask.type() == 'torch.FloatTensor'


def test_rgba_to_rgb():
    transform = tvt.Lambda(rgba_to_rgb)
    ign_ds = make_kasmi_ign_dataset(img_transforms=transform)
    loader = DataLoader(ign_ds)
    for img, _ in loader:
        assert img.shape == (1, 3, 400, 400)
        assert img.type() == 'torch.ByteTensor'


def test_composed_transforms():
    img_transform = tvt.Compose([tvt.Resize((256, 256))])
    mask_transform = tvt.Compose([tvt.Resize((256, 256)), tvt.Lambda(one_hot)])
    hou_ds = make_hou_loader(
        HOU_CROPLAND_DIR,
        img_transform=img_transform,
        mask_transform=mask_transform
    )
    for img, mask in hou_ds:
        assert img.shape == (1, 3, 256, 256)
        assert mask.shape == (1, 2, 256, 256)
        n_unique = len(mask.unique())
        assert n_unique == 2
         # Pixel values in `mask` where there are solar
        # panels must be one.
        assert mask.max() == 1
        # Pixel values in `mask` where there are no solar
        # panels must be zero.
        assert mask.min() == 0
        # Test type:
        # - The image should be uint8 or ByteTenor for efficiency.
        # - The mask must be a float for the loss calculation during
        #   training.
        assert img.type() == 'torch.ByteTensor'
        assert mask.type() == 'torch.FloatTensor'
