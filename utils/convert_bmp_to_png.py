"""PyTorch torchvision only supports PNG or JPEG image file formats.
 So, this module converts all the BMP files (which are in the Hou
 dataset) into PNG format, and deletes the old BMP files."""


from pathlib import Path

from PIL import Image


def convert_bmp2png(root: Path) -> None:
    """Convert all BMP files in a directory and its subdirectories
    int PNG format."""
    total = _count_files(root)
    count = 0
    for bmp in root.glob('**/*.bmp'):
        png = f'{str(bmp)[:-3]}png'
        if not Path(png).exists():
            Image.open(bmp).save(png)
        count += 1
        _display_progress(count, total)


def _count_files(root: Path) -> int:
    total = 0
    for _ in root.glob('**/*.bmp'):
        total += 1
    return total


def _display_progress(count: int, total: int) -> None:
    if count % 100 == 0:
        count_ = count // 100
        diff = (total // 100) - count_        
        print(
            f"\rProgress: [{'='*(count_-1)}>{'.'*diff}] ({(count*100)//total}%)",
            end='',
            flush=True
        )


def delete_bmp_files(root: Path) -> None:
    """Delete all BMP files in a directory and its subdirectories."""
    total = _count_files(root)
    count = 0
    for img in root.glob('**/*.bmp'):
        img.unlink()
        count += 1
        _display_progress(count, total)


if __name__ == '__main__':
    root = Path('data/Hou')
    print('Convert BMP files to PNG:')
    convert_bmp2png(root)
    print('\nDelete BMP files')
    delete_bmp_files(root)
