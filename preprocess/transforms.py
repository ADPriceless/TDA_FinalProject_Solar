"""Preprocessing functions to apply to images in the dataset.
The functions should be supplied to a 
`torchvision.transforms.Lambda` class.
"""


import torch


def one_hot(mask: torch.Tensor) -> torch.Tensor:
    """One-hot encode the mask along dimension 1, where the mask
    shape is `[batch_size x n_classes x h x w]`. Currently only
    supports binary masks (2 classes)."""
    return torch.concat((mask == 0, mask > 0), dim=0).float()


def rgba_to_rgb(img: torch.Tensor) -> torch.Tensor:
    """Convert RGBA image to RGB image."""
    return img[:3, :, :]
