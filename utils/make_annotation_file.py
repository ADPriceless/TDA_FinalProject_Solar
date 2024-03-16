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
