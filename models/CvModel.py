import numpy as np
from scipy.stats import mode

class CvModel:

    def __init__(self, n_folds, trans, clf):
        self.trans = trans
        self.clf = clf
        self.n_folds = n_folds
        self.models = None

    def fit(self, X, y):

        X = self.trans.transform(X)

        models = []

        fold_size = X.shape[0] / self.n_folds
        for i in range(self.n_folds):
            X_fold = np.concatenate([X[:i*fold_size], X[(i+1)*fold_size:]])
            y_fold = np.concatenate([y[:i*fold_size], y[(i+1)*fold_size:]])
            models.append(self.clf.fit(X_fold, y_fold))

        self.models = models

    def predict(self, X):

        X = self.trans.transform(X)
        votes = np.array([ m.predict(X) for m in self.models ])
        return (np.mean(votes, axis=0) >= 0.5).astype(int)

    def hill_predict(self, X):

        X = self.trans.transform(X)

        predictions = []

        fold_size = X.shape[0] / self.n_folds
        for i in range(self.n_folds):
            m, X_fold = self.models[i], X[i*fold_size:(i+1)*fold_size]
            predictions.append(m.predict(X_fold))

        return np.concatenate(predictions).astype(int)


if __name__ == '__main__':

    from pyutils.kaggle_io.extract_inputs import extract_training_data, extract_testing_data

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.externals import joblib

    print 'Reading data...'
    Id_train, X_train, y_train = extract_training_data('data/kaggle_train_tf_idf.csv')
    Id_test, X_test = extract_testing_data('data/kaggle_test_tf_idf.csv')

    print 'Initializing CvModel...'
    n_folds = 5
    scaler = StandardScaler().fit(X_train)
    rf = RandomForestClassifier(
        n_estimators=100,
        max_features='auto'
    )
    cv_rf = CvModel(n_folds, scaler, rf)

    print 'Training CvModel...'
    cv_rf.fit(X_train, y_train)

    print 'Pickling...'
    joblib.dump(cv_rf, 'random_forests/rf1.pkl')

    print 'UnPickling...'
    cv_rf_pkl = joblib.load('random_forests/rf1.pkl')

    print 'Check hill_predict of pickle and unpickled CvModel... (should have single True)'
    print np.unique(cv_rf_pkl.hill_predict(X_train) == cv_rf.hill_predict(X_train))

    print 'Check predict of pickle and unpickled CvModel... (should have single True)'
    print np.unique(cv_rf_pkl.predict(X_test) == cv_rf.predict(X_test))

