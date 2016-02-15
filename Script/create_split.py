############################################
# Create a split of three dataframes train/val/test
# and store them in csv files in train-val-test folder
# Todo once at the beginning of time !!!!
############################################


import os
from os.path import expanduser
home = expanduser("~")
datap = os.path.join('Documents', 'project',
                     'kagle', 'yelp', 'Data', 'raw')

import pandas as pd
import numpy as np


def detect(i, labels):
    if str(i) in labels.split(' '):
        return 1
    else:
        return 0


def generate_train_val_test(path, train_photos):
    df = train_photos
    df = df.reindex(np.random.permutation(df.index))
    N = len(df)
    # Split for a test of 25%
    size_test = int(0.25 * N)
    test = df.iloc[:size_test]
    train = df.iloc[size_test:]
    train.index = range(len(train))
    # Split for a train/Val
    size_val = int(0.25 * len(train))
    val = train.iloc[:size_val]
    train = train.iloc[size_val:]
    train.index = range(len(train))

    # Grab the file on disk
    istrain = os.path.isfile(os.path.join(path, 'train.csv'))
    isval = os.path.isfile(os.path.join(path, 'val.csv'))
    istest = os.path.isfile(os.path.join(path, 'test.csv'))
    if istrain and isval and istest:
        raise ValueError(
            'ATTENTION: des fichiers train/val/test existe deja!!!!!!! Remove them before going further if you wish !!!')

    train.to_csv(os.path.join(path, 'train.csv'), index=None)
    test.to_csv(os.path.join(path, 'test.csv'), index=None)
    val.to_csv(os.path.join(path, 'val.csv'), index=None)


def load_df_train_val():
    train = pd.read_csv(os.path.join(path, 'train.csv'))
    val = pd.read_csv(os.path.join(path, 'val.csv'))
    return train, val


if __name__ == "__main__":

    train = pd.read_csv(os.path.join(home, datap, 'train.csv'))
    train_photo_to_biz_ids = pd.read_csv(
        os.path.join(home, datap, 'train_photo_to_biz_ids.csv'))
    train_photo_to_biz_ids.photo_id = map(str, train_photo_to_biz_ids.photo_id)
    test_photo_to_biz = pd.read_csv(
        os.path.join(home, datap, 'test_photo_to_biz.csv'))

    train_photos_names = os.listdir(os.path.join(home, datap, 'train_photos'))
    train_photos_paths = [os.path.join(datap, 'train_photos', f)
                          for f in train_photos_names]
    train_photos_ids = [int(f.split('.')[0]) for f in train_photos_names]
    train_photos = pd.DataFrame(
        np.array([train_photos_names, train_photos_paths, train_photos_ids]).T,
        columns=['photo_name', 'photo_path', 'photo_id'])
    train_photos = pd.merge(
        train_photos, train_photo_to_biz_ids, on='photo_id')
    train_photos = pd.merge(train_photos, train, on='business_id')
    train_photos.columns = list(train_photos.columns[:-1]) + ['labs']
    train_photos = train_photos.dropna()
    train_photos.index = range(len(train_photos))

    val = [map(lambda x:detect(i, x), train_photos.labs) for i in range(9)]
    df = pd.DataFrame(np.array(val).T, columns=[
        'label_' + str(i) for i in range(9)])
    train_photos = train_photos.join(df)

    path = os.path.join(home, 'Documents', 'project', 'kagle',
                        'yelp', 'Data', 'train-val-test')

    generate_train_val_test(path, train_photos)
