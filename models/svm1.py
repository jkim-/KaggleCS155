#! /nfs/raid13/babar/software/anaconda/bin/python

import sys
from os import path
sys.path.append('/nfs/raid13/babar/dchao/KaggleCS155')

from pyutils.kaggle_io.extract_inputs import extract_training_data
from pyutils.ensemble_selection.CvModel import CvModel
from sklearn.externals import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('C',
                        help='Soft margin.')
    parser.add_argument('gamma',
                        help='Rbf width')
    parser.add_argument('model_fname',
                        help='Absolute path to pickle the fitted CvModel.')
    args = parser.parse_args()

    Id, X, y = extract_training_data('/nfs/raid13/babar/dchao/KaggleCS155/data/kaggle_train_tf_idf.csv')

    # trans/clf specs
    n_folds = 5
    scaler = MinMaxScaler(feature_range=(-1,1)).fit(X)
    clf = SVC(
        C=float(args.C),
        gamma=float(args.gamma)
    )

    cv_clf = CvModel(n_folds, scaler, clf)
    cv_clf.fit(X, y)
    joblib.dump(cv_clf, args.model_fname)
