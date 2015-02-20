#! /nfs/raid13/babar/software/anaconda/bin/python

import sys
sys.path.append('/nfs/raid13/babar/dchao/KaggleCS155')

from pyutils.kaggle_io.extract_inputs import extract_training_data
from pyutils.ensemble_selection.CvModel import CvModel
from sklearn.externals import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('penalty',
                        help='L1 or L2 penalization.')
    parser.add_argument('C',
                        help='Regularization parameter.')
    parser.add_argument('fit_intercept',
                        help='Whether to include a constant bias term in the loss function.')
    parser.add_argument('n_components_pca',
                        help='n_components for PCA.')
    parser.add_argument('model_fname',
                        help='Absolute path to pickle the fitted CvModel.')
    args = parser.parse_args()

    n_components_pca = None if args.n_components_pca == 'None' else int(args.n_components_pca)

    Id, X, y = extract_training_data('/nfs/raid13/babar/dchao/KaggleCS155/data/kaggle_train_tf_idf.csv')

    # trans/clf specs
    n_folds = 5
    scaler = StandardScaler()
    pca = PCA(n_components=n_components_pca, whiten=True)
    trans = Pipeline([('scale_center', scaler),('pca', pca)]).fit(X)
    clf = LogisticRegression(
        penalty = args.penalty,
        C = float(args.C),
        fit_intercept = bool(args.fit_intercept),
        class_weight = 'auto'
    )

    cv_clf = CvModel(n_folds, trans, clf)
    cv_clf.fit(X, y)
    joblib.dump(cv_clf, args.model_fname)
