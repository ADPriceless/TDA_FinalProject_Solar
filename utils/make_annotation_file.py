"""This utility file makes annotation CSV files used by the custom 
`Dataset`s and `DataLoader`s for the collected datasets."""


import csv
from pathlib import Path


def create_annotation_files_hou(
        annotation_path: Path, root_dir: Path,
        exclude_dirs: list[Path] = None, file_ext: str = 'png'
) -> None:
    """Writes the paths of the images and masks, relative to 
    `root_dir`, to a CSV file."""
    if annotation_path.exists():
        return
    filepath_generator = root_dir.glob(f'**/*.{file_ext}')
    while True:
        try:
            img_filepath, mask_filepath = get_img_and_mask_filepaths(filepath_generator)
            # Prevent files in excluded directories from being added to
            # annotation.csv
            if exclude_dirs is not None:
                if any((
                    path_excluded(img_filepath, exclude_dirs),
                    path_excluded(mask_filepath, exclude_dirs),
                )):
                    continue
            img_filepath.relative_to(root_dir)
            write_path_of_img_and_mask_to_csv(
                img_filepath.relative_to(root_dir),
                mask_filepath.relative_to(root_dir),
                annotation_path
            )
        except StopIteration:
            break


def get_img_and_mask_filepaths(generator) -> tuple[Path, Path]:
    """Returns a tuple containing the image and mask filepaths."""
    img_filepath = next(generator)
    mask_filepath = next(generator)
    return img_filepath, mask_filepath


def path_excluded(filepath: Path, exclude_dirs: list[Path]) -> bool:
    """Returns True if the given filepath is in one of the given directories."""
    for directory in exclude_dirs:
        if filepath.is_relative_to(directory):
            return True
    return False


def write_path_of_img_and_mask_to_csv(
    img_filepath: Path, mask_filepath: Path, csv_filepath: Path
) -> None:
    """Write the paths of the images and masks, relative to 
    `root_dir`, to a CSV file."""
    with open(csv_filepath, 'a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([img_filepath, mask_filepath])


def create_annotation_file_kasmi(img_root: Path, mask_root: Path, csv_path: Path) -> None:
    """Writes the images and their masks to a CSV file"""
    _new_annotation_file(csv_path)
    blank_mask_path = Path('data/Kasmi/bdappv/blank_mask.png')
    img_gen = img_root.glob('*.png')
    mask_gen = mask_root.glob('*.png')
    # For each mask, there is an image. But for images without solar
    # panels, there is no mask.
    # So, loop through masks and try to match them to an image. If
    # the image doesn't match, pair it with the blank mask.
    # Once the masks have run out, any images that are leftover are
    # not included in the annotation file.
    img_path = next(img_gen) # get the first image
    while True:
        try:
            mask_path = next(mask_gen)
            while img_path.name != mask_path.name:
                write_path_of_img_and_mask_to_csv(img_path, blank_mask_path, csv_path)
                img_path = next(img_gen)
            write_path_of_img_and_mask_to_csv(img_path, mask_path, csv_path)
            img_path = next(img_gen)
        except StopIteration:
            break # Ran out of masks


def _new_annotation_file(csv_path: Path) -> None:
    with open(csv_path, 'w', newline='', encoding='utf-8'):
        pass # CSV is empty
