"""This file contains class representations of the datasets
that have been collected for this project."""


from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class HouDataset(Dataset):
    """Class representing the Hou dataset. """
    def __init__(
        self,
        annotations_file: Path,
        img_dir: Path,
        transform=None,
        target_transform=None
    ):
        self.df_img_mask = pd.read_csv(annotations_file, names=['img', 'mask'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.df_img_mask.shape[0]

    def __getitem__(self, idx):
        img_path = self.img_dir.joinpath(self.df_img_mask.loc[idx, 'img'])
        mask_path = self.img_dir.joinpath(self.df_img_mask.loc[idx,'mask'])
        image = read_image(str(img_path))
        mask = read_image(str(mask_path))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask
