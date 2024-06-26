"""This file contains class representations of the datasets
that have been collected for this project."""


from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

from utils.make_annotation_file import create_annotation_files_hou, create_annotation_file_kasmi


def make_hou_dataset(
    exclude_dirs: list[Path],
    img_transforms=None,
    mask_transforms=None
) -> Dataset:
    """Make a dataset class for the Hou dataset."""
    hou_root = Path('data/hou')
    if not hou_root.exists():
        raise ValueError(f'hou_root does not exist: {hou_root}')
    annotaion_file = hou_root.joinpath('annotation.csv')
    create_annotation_files_hou(annotaion_file, hou_root, exclude_dirs)
    return ImgAndMaskDataset(annotaion_file, hou_root, img_transforms, mask_transforms)


def make_kasmi_ign_dataset(img_transforms=None, mask_transforms=None) -> Dataset:
    """Make a dataset class for the Kasmi ign dataset."""
    ign_root = Path('data/kasmi/bdappv/ign')
    annotation_path = ign_root.joinpath('annotation.csv')
    create_annotation_file_kasmi(annotation_path, ign_root)
    return ImgAndMaskDataset(annotation_path, ign_root, img_transforms, mask_transforms)


class ImgAndMaskDataset(Dataset):
    """Class representing a dataset containing images and masks
    for segmentation tasks."""
    def __init__(
        self,
        annotations_file: Path,
        img_dir: Path,
        transform=None,
        target_transform=None,
    ):
        if img_dir.is_file():
            raise ValueError(f'img_dir must be a directory, not a file: {img_dir}')
        self.df_img_mask = pd.read_csv(annotations_file, names=['img', 'mask'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_path = None
        self.mask_path = None

    def __len__(self):
        return self.df_img_mask.shape[0]

    def __getitem__(self, idx):
        self.img_path = self.df_img_mask.loc[idx, 'img']
        self.mask_path = self.df_img_mask.loc[idx,'mask']
        image = read_image(self.img_path)
        mask = read_image(self.mask_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask
