#! /nfs/raid13/babar/software/anaconda/bin/python

import sys
from os import path
sys.path.append('/nfs/raid13/babar/dchao/KaggleCS155')

import numpy as np
from sklearn.externals import joblib

from pyutils.kaggle_io.extract_inputs import extract_training_data, extract_testing_data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_folder',
                        help='Model folder to test. Sould contain 1.pkl.')
    parser.add_argument('model_index', default='1',
                        help='Model number to test.')
    args = parser.parse_args()

    Id, X, y = extract_training_data('/nfs/raid13/babar/dchao/KaggleCS155/data/kaggle_train_tf_idf.csv')

    cv_rf = joblib.load(args.model_folder + '/' + args.model_index + '.pkl')

    print np.unique(cv_rf.hill_predict(X))
