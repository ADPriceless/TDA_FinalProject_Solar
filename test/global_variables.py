"""Global variables (constants) that are common to multiple tests"""

from pathlib import Path


DATA_DIR = Path('data/Hou/PV01_Rooftop_Brick')

DIRS_UNDER_TEST = [p for p in Path('data/Hou').iterdir() if p.is_dir()]

_HOU_DIR_LENGTHS = (138, 413, 94, 859, 117, 352, 119, 625, 236)
DS_SUBSET_LENGTHS = \
    dict(zip([dir_.parts[-1] for dir_ in DIRS_UNDER_TEST], _HOU_DIR_LENGTHS))


if __name__ == '__main__':
    # print all variables
    print('DATA_DIR:')
    print(DATA_DIR)

    print('DIRS_UNDER_TEST:')
    print('\n'.join([str(p) for p in DIRS_UNDER_TEST]))
    for p in DIRS_UNDER_TEST:
        assert p.is_dir()
    print()

    print('DS_SUBSET_LENGTHS:')
    print('\n'.join([f'{k}: {v}' for k, v in DS_SUBSET_LENGTHS.items()]))
    print()
