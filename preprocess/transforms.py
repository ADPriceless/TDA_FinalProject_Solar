"""Preprocessing functions to apply to images in the dataset.
The functions should be supplied to a 
`torchvision.transforms.Lambda` class.
"""


import torch


def scale_pixels(image: torch.Tensor) -> torch.Tensor:
    """Rescale pixel values to range [0, 1]"""
    return image / 255


def binarise_mask(mask: torch.Tensor) -> torch.Tensor:
    """Converts pixel values in target to 0 and 1 only."""
    return (mask > 0).float()
