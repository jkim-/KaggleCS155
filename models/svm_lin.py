#! /nfs/raid13/babar/software/anaconda/bin/python

import sys
sys.path.append('/nfs/raid13/babar/dchao/KaggleCS155')

from pyutils.kaggle_io.extract_inputs import extract_training_data
from pyutils.ensemble_selection.CvModel import CvModel
from sklearn.externals import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('C',
                        help='C the regularization parameter for svm.')
    parser.add_argument('kernel',
                        help='kernel for svm.')
    parser.add_argument('n_components_pca',
                        help='n_components for pca.')
    parser.add_argument('whiten',
                        help='whiten for pca.')
    parser.add_argument('model_fname',
                        help='Absolute path to pickle the fitted CvModel.')
    args = parser.parse_args()

    n_components_pca = None if args.n_components_pca == 'None' else int(args.n_components_pca)

    Id, X, y = extract_training_data('/nfs/raid13/babar/dchao/KaggleCS155/data/kaggle_train_tf_idf.csv')

    # trans/clf specs
    n_folds = 5
    scaler = MinMaxScaler(feature_range=(-1.,1.))
    pca = PCA(n_components=n_components_pca, whiten=bool(args.whiten))
    trans = Pipeline([('scale_center', scaler),('pca', pca)]).fit(X)
    clf = SVC(
        C=float(args.C),
        kernel=args.kernel
    )

    cv_clf = CvModel(n_folds, trans, clf)
    cv_clf.fit(X, y)
    joblib.dump(cv_clf, args.model_fname)
