"""Copy kddtrain.csv and kddtest.csv into this project folder.

Run from repository root (Windows PowerShell / cmd):

python mini_ids_project\copy_data.py

This script will copy `kddtrain.csv` and `kddtest.csv` from the repository root
into `mini_ids_project/` (overwriting if present).
"""
import shutil
import os

SRC_TRAIN = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'kddtrain.csv')
SRC_TEST = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'kddtest.csv')
DST_DIR = os.path.dirname(__file__)


def main():
    print('Source train:', SRC_TRAIN)
    print('Source test :', SRC_TEST)

    if not os.path.exists(SRC_TRAIN):
        print('ERROR: source train file not found at', SRC_TRAIN)
        return
    if not os.path.exists(SRC_TEST):
        print('ERROR: source test file not found at', SRC_TEST)
        return

    dst_train = os.path.join(DST_DIR, 'kddtrain.csv')
    dst_test = os.path.join(DST_DIR, 'kddtest.csv')

    shutil.copy2(SRC_TRAIN, dst_train)
    shutil.copy2(SRC_TEST, dst_test)

    print('Copied to', dst_train)
    print('Copied to', dst_test)


if __name__ == '__main__':
    main()
