import numpy as np
from scipy.stats import mode

class CvModel:

    def __init__(self, n_folds, trans, clf):
        self.trans = trans
        self.clf = clf
        self.n_folds = n_folds
        self.fold_demarcations = None
        self.models = None

    def fit(self, X, y):

        X = self.trans.transform(X)

        models = []

        fold_size = X.shape[0] / self.n_folds
        for i in range(self.n_folds):
            X_fold = X[i*fold_size : (i+1)*fold_size]
            y_fold = y[i*fold_size : (i+1)*fold_size]
            models.append(self.clf.fit(X_fold, y_fold))

        self.fold_demarcations = np.arange(fold_size, X.shape[0] + 1, fold_size)
        self.models = models

    def predict(self, x):
        x = self.trans.transform(X)
        votes = np.array([ m.predict(x)[0] for m in self.models ], dtype=np.int8)
        return mode(votes)[0][0]

    def hill_predict(self, x, sample_num):

        lo = np.flatnonzero(self.fold_demarcations / (sample_num + 1))[0]
        models = self.models[:lo] + self.models[lo+1:]

        x = self.trans.transform(x)
        votes = np.array([ m.predict(x)[0] for m in models ], dtype=np.int8)

        return mode(votes)[0][0]

if __name__ == '__main__':

    from pyutils.kaggle_io.extract_inputs import extract_training_data

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.externals import joblib

    print 'Reading data...'
    Id, X, Y = extract_training_data('data/kaggle_train_tf_idf.csv')
    scaler = StandardScaler().fit(X)

    print 'Initializing CvModel...'
    rf = RandomForestClassifier(
        n_estimators=100,
        max_features='auto'
    )

    cv_rf = CvModel(5, scaler, rf)

    print 'Training CvModel...'
    cv_rf.fit(X, Y)

    print 'Pickling...'
    joblib.dump(cv_rf, 'random_forests/rf1.pkl')

    print 'UnPickling...'
    cv_rf_pkl = joblib.load('random_forests/rf1.pkl')

    print 'Check hill_predict of pickle and unpickled CvModel...'
    print cv_rf_pkl.hill_predict(X[909], 909) == cv_rf.hill_predict(X[909], 909)

    print cv_rf_pkl.models
