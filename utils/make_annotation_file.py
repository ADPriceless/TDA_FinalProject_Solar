"""This utility file makes annotation CSV files used by the custom 
`Dataset`s and `DataLoader`s for the collected datasets."""


import csv
from pathlib import Path


def write_img_and_mask_names_to_csv(
        root_dir: Path, csv_file_path: Path, file_ext: str = 'bmp'
) -> None:
    """Writes the names of the images and masks to a CSV file for use in
    data retrieval."""
    img_or_mask = 0
    for filename in root_dir.glob(f'*.{file_ext}'):
        if img_or_mask == 0:
            img_file = filename.name
            img_or_mask = 1
        else:
            # write the `img_file` and `filename` (mask file) to csv
            with open(csv_file_path, 'a', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([img_file, filename.name])
            img_or_mask = 0


def write_annotation_files(directories: list[Path]):
    """Writes the annotation files for the given directories."""
    for directory in directories:
        annotation_path = directory.joinpath('annotation.csv')
        if not annotation_path.exists():
            write_img_and_mask_names_to_csv(
                directory,
                annotation_path
            )


if __name__ == '__main__':
    root = Path('data/Hou')
    directories = list(root.iterdir())
    write_annotation_files(directories)
