"""Make a blank mask since they are not included in the Kasmi dataset."""

from pathlib import Path

from PIL import Image


def make_blank_mask(filepath: Path, size: tuple[int, int]) -> None:
    """Create a black PNG image"""
    image = Image.new(mode='L', size=size, color=0)
    image.save(filepath)


if __name__ == '__main__':
    path = Path('data/Kasmi/bdappv/blank_mask.png')
    make_blank_mask(path, (520, 520))
