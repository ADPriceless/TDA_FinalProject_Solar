"""Functions to load datasets"""


from pathlib import Path

from torch.utils.data import DataLoader

from preprocess.datasets import ImgAndMaskDataset
from utils.make_annotation_file import create_annotation_files_hou


def make_hou_loader(
    ds_root: Path,
    exclude_dirs: list[Path] = None,
    img_transform = None,
    mask_transform = None,
    params: dict = None
) -> DataLoader:
    """
    ## Description
    Load the Hou dataset as image/mask pairs.
    
    ---

    ## Data Structure
    - This dataset consists of several subdirectories which state where the 
    solar panels are setup (e.g. rooftop, grassland, etc.)
    - Within each subdirectory, the data are already in image/mask pairs
    """
    _check_types(ds_root, exclude_dirs, params)
    annotation_file = ds_root.joinpath('annotation.csv')
    create_annotation_files_hou(annotation_file, ds_root, exclude_dirs)
    dataset = ImgAndMaskDataset(
        annotation_file,
        ds_root,
        transform=img_transform,
        target_transform=mask_transform
    )
    if params is not None:
        return DataLoader(dataset, **params)
    return DataLoader(dataset)


def _check_types(ds_root, exclude_dirs, params):
    if not isinstance(ds_root, Path):
        raise TypeError('ds_root must be a Path object')
    # check exclude_dirs is iterable
    if not exclude_dirs is None:
        try:
            for dir_ in exclude_dirs:
                if not isinstance(dir_, Path):
                    raise TypeError('exclude_dirs must be a list of Path objects')
        except TypeError as err:
            raise TypeError('exclude_dirs must be a list of Path objects') from err
    if not (params is None or isinstance(params, dict)):
        raise TypeError('params must be a dictionary')


def make_kasmi_loader(ds_root: Path):
    """
    ## Description
    Load the Kasmi crowd-source dataset as image/mask pairs.
    
    ---

    ## Data Structure
    - This dataset consists of a folder for the images and one for the mask.
    - The filename of the image its corresponding mask are the same
    - An image may contain no solar panels. In this case, there is no mask
    (so the function generates a blank mask)
    """
    raise NotImplementedError('Will implement this after the Hou dataset,' \
                              ' since that one seems simpler to do.')
