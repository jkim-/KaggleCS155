#! /nfs/raid13/babar/software/anaconda/bin/python

import sys
sys.path.append('/nfs/raid13/babar/dchao/KaggleCS155')

from pyutils.kaggle_io.extract_inputs import extract_training_data
from pyutils.ensemble_selection.CvModel import CvModel
from sklearn.externals import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('n_estimators',
                        help='n_estimators of AdaBoostClassifier.')
    parser.add_argument('max_depth',
                        help='max_depth of base estimator DecisionTreeClassifier.')
    parser.add_argument('learning_rate',
                        help='learning_rate of each tree.')
    parser.add_argument('model_fname',
                        help='Absolute path to pickle the fitted CvModel.')
    args = parser.parse_args()

    Id, X, y = extract_training_data('/nfs/raid13/babar/dchao/KaggleCS155/data/kaggle_train_tf_idf.csv')

    # trans/clf specs
    n_folds = 5
    scaler = StandardScaler().fit(X)
    dtclf = DecisionTreeClassifier(max_depth=int(args.max_depth))
    clf = AdaBoostClassifier(
        base_estimator=dtclf,
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate)
    )

    cv_clf = CvModel(n_folds, scaler, clf)
    cv_clf.fit(X, y, [1. if yi == 0 else 3. for yi in y])
    joblib.dump(cv_clf, args.model_fname)
