"""
Create cached images to memmap
"""
from __future__ import division
import argparse
import os
import sys
from time import strftime
import pandas as pd
import numpy as np
from skimage.io import imread
from tqdm import tqdm
from skimage.transform import resize

from os.path import expanduser
home = expanduser("~")
datap = os.path.join(home, 'Documents', 'project',
                     'kagle', 'yelp', 'Data')


def check_if_image_exists(fname):
    return os.path.exists(fname)


def get_current_date():
    return strftime('%Y%m%d')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--whichset', required=True, type=str,
                        help='train/test or val ?')
    parser.add_argument('--size', required=True, type=int,
                        help='Size of the image')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing cache')

    args = parser.parse_args()

    dfname = os.path.join(datap, 'train-val-test', args.whichset + '.csv')
    df = pd.read_csv(dfname)

    df['exist'] = df['photo_path'].apply(check_if_image_exists)
    print '%i does not exists' % (len(df) - df['exist'].sum())
    print df[~df['exist']]

    df = df[df['exist']]
    df = df.reset_index(drop=True)

    if not os.path.isdir(os.path.join(datap, 'train-val-test', 'cache')):
        os.mkdir(os.path.join(datap, 'train-val-test', 'cache'))

    X_fname = 'cache/X_%s_%s_%s.npy' % (args.whichset, args.size,
                                        get_current_date())

    y_fname = 'cache/y_%s_%s_%s.npy' % (args.whichset, args.size,
                                        get_current_date())

    X_shape = (len(df), 3, args.size, args.size)
    y_shape = (len(df), 9)

    print X_shape, y_shape

    if os.path.exists(os.path.join(datap, 'train-val-test', X_fname)) and not args.overwrite:
        print '%s exists. Use --overwrite' % X_fname
        sys.exit(1)

    if os.path.exists(os.path.join(datap, 'train-val-test', y_fname)) and not args.overwrite:
        print '%s exists. Use --overwrite' % y_fname
        sys.exit(1)

    print 'Will write X to %s with shape of %s' % (X_fname, X_shape)
    print 'Will write y to %s with shape of %s' % (y_fname, y_shape)

    X_fp = np.memmap(os.path.join(datap, 'train-val-test', X_fname),
                     dtype=np.float32, mode='w+', shape=X_shape)
    y_fp = np.memmap(os.path.join(datap, 'train-val-test', y_fname),
                     dtype=np.int32, mode='w+', shape=y_shape)

    labs = [f for f in df.columns if f.split('_')[0] == 'label']

    print 'Processing begin'
    for i, row in tqdm(df.iterrows(), total=len(df)):
        fname = row.photo_path
        labels = row[labs]

        try:
            img = imread(fname)
            img_h, img_w, _ = img.shape
            img = resize(img, (args.size, args.size))
            img = img.transpose(2, 0, 1)
            img = img.astype(np.float32)

            assert img.shape == (3, args.size, args.size)
            assert img.dtype == np.float32

            X_fp[i] = img
            y_fp[i] = labels

            X_fp.flush()
            y_fp.flush()

        except:
            print '%s has failed' % i
