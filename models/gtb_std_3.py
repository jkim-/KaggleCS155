#! /nfs/raid13/babar/software/anaconda/bin/python

import sys
sys.path.append('/nfs/raid13/babar/dchao/KaggleCS155')

from pyutils.kaggle_io.extract_inputs import extract_training_data
from pyutils.ensemble_selection.CvModel import CvModel
from sklearn.externals import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('loss',
                        help='Loss funtion to be optimized.')
    parser.add_argument('learning_rate',
                        help='Learning rate for each tree.')
    parser.add_argument('n_estimators',
                        help='Rounds of boosting.')
    parser.add_argument('max_depth',
                        help='max_depth for each tree.')
    parser.add_argument('model_fname',
                        help='Absolute path to pickle the fitted CvModel.')
    args = parser.parse_args()

    Id, X, y = extract_training_data('/nfs/raid13/babar/dchao/KaggleCS155/data/kaggle_train_tf_idf.csv')

    # trans/clf specs
    n_folds = 5
    scaler = StandardScaler()
    pca = PCA(n_components=None, whiten=True)
    trans = Pipeline([('center_scale', scaler),('whiten', pca)]).fit(X)
    clf = GradientBoostingClassifier(
        loss=args.loss,
        learning_rate=float(args.learning_rate),
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth)
    )

    cv_clf = CvModel(n_folds, trans, clf)
    cv_clf.fit(X, y)
    joblib.dump(cv_clf, args.model_fname)
