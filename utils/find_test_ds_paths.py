"""This file creates a CSV of the filepaths of the images and masks
which comprise a chosen test set."""

import warnings
import math
from pathlib import Path
import shutil
from typing import (
    List,
    Sequence,
    TypeVar,
    Union,
)

import torch
from torch import Generator
# No 'default_generator' in torch/__init__.pyi
from torch import randperm # pylint: disable=no-name-in-module
from torch.utils.data import Dataset, Subset
import torchvision.transforms.v2 as tv_transforms

from preprocess.datasets import make_hou_dataset, ImgAndMaskDataset
from preprocess import transforms as custom_transforms


T = TypeVar('T')


def random_split_edited(dataset: Dataset[T], lengths: Sequence[Union[int, float]],
                 generator: Generator) -> tuple[List[Subset[T]], list]:
    r"""
    Modified version of PyTorch's `torch.utils.data.random_split` function. The
    function now returns a tuple of the lengths of each of the subsets and the
    randomly shufflied indicies of the source dataset.

    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    Example:
        >>> # xdoctest: +SKIP
        >>> generator1 = torch.Generator().manual_seed(42)
        >>> generator2 = torch.Generator().manual_seed(42)
        >>> random_split(range(10), [3, 7], generator=generator1)
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[arg-type, call-overload]
    return lengths, indices


def copy_dataset_files(dataset: ImgAndMaskDataset, indices: list[int], dest: str) -> None:
    """Copy the filepaths of a dataset to `dest`, from given indices."""
    for i in indices:
        # Internally set the filepath attributes of the
        # dataset by calling its `__getitem__` method.
        _ = dataset[i]
        shutil.copy(dataset.img_path, dest + dataset.img_path.split('\\')[-1])
        shutil.copy(dataset.mask_path, dest + dataset.mask_path.split('\\')[-1])


def main():
    RANDOM_STATE = 42
    rng = torch.Generator().manual_seed(RANDOM_STATE)

    output_h_w = [520, 520]
    mask_transforms = tv_transforms.Compose(
        [
            tv_transforms.Resize(output_h_w),
            tv_transforms.Lambda(custom_transforms.one_hot),
        ]
    )

    # Excluding this directory because the masks appear to be wrong
    EXCLUDE_DIRS = (Path('data/Hou/PV03_Ground_WaterSurface'),)
    hou_ds = make_hou_dataset(
        EXCLUDE_DIRS,
        # img_transforms=input_transforms,
        mask_transforms=mask_transforms,
    )

    # Only interested in the indices generated from the test dataset
    (_, _, test_ds_len), indices = random_split_edited(hou_ds, [0.6, 0.2, 0.2], generator=rng)
    test_indices = indices[-test_ds_len:]

    test_ds_path = 'data/test_ds/'

    copy_dataset_files(hou_ds, test_indices, test_ds_path)


if __name__ == '__main__':
    main()
