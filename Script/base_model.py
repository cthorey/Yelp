from __future__ import print_function

import sys
import os
import time
from os.path import expanduser
home = expanduser("~")

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

import lasagne


datap = os.path.join(home, 'Documents', 'project',
                     'kagle', 'yelp', 'Data', 'raw')


def load_dataset():

    train = pd.read_csv(os.path.join(datap, 'train.csv'))
    val = pd.read_csv(os.path.join(datap, 'val.csv'))
    test = pd.read_csv(os.path.join(datap, 'test.csv'))
