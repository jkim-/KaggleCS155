import numpy as np
from scipy.stats import mode

# Class for a cross validated models used in ensemble selection.
class CvModel:
    """
    This is the class for a cross validated model as discussed in
    Caruana's paper.

    Example Usage:

        # To instantiate a model, input the following:
        # nfolds: The number of cross-validation folds.
        # trans: Input preprocessor object. It should have a tranform method.
        # clf: A classifier template that has a fit and predict method.
        nfolds, trans, clf = 5, StandardScaler(), RandomForestClassifier()
        cv_model = CvModel(n_folds, trans, clf)

        # Fit the model with training data X and labels y.
        cv_model.fit(X_train, y_train)

        # Predict in sample labels. Note that this method is only valid for the
        # same X_train passed into the fit method.
        cv_model.hill_predict(X_train)

        # Predict out of sample labels. This could be any X_test.
        cv_model.predict(X_test)
    """

    def __init__(self, n_folds, trans, clf):
        self.trans = trans
        self.clf = clf
        self.n_folds = n_folds
        self.models = None

    # Fit the model.
    def fit(self, X, y):
        """
        Train the cross-validated model using the training data X and y.

        Input:

            X: ndarray of shape (n_training_examples, # features).
               This is the training design matrix.

            y: ndarray of shape (n_training_examples, ).
               These are the training taret labels.

        Output/Side effect:

            self.models gets populated with a list of self.n_folds models.
            The ith model in the list is the model trained using all but
            the ith fold of the training data.

            Returns nothing.

        """

        # Input preprocessing.
        X = self.trans.transform(X)

        # Initialize the cross-validated models list.
        models = []

        # Train each team member.
        fold_size = X.shape[0] / self.n_folds
        for i in range(self.n_folds):

            # Get the folds to train on. Leave the ith fold out of training.
            X_fold = np.concatenate([X[:i*fold_size], X[(i+1)*fold_size:]])
            y_fold = np.concatenate([y[:i*fold_size], y[(i+1)*fold_size:]])

            # Train using the cached classifier prototype.
            models.append(self.clf.fit(X_fold, y_fold))

        self.models = models

    # Predict on unseen/test data.
    def predict(self, X):
        """
        Predict unseen/test data.

        Input:

            X: ndarray of shape (n_data_points, n_features).
               This is the unseen data.

        Output:

            Predicted class labels in the form of an int ndarray
            of shape (n_data_points,). It is either 0 or 1.
        """

        # Input preprocessing.
        X = self.trans.transform(X)

        # All models in the team cast their vote for every point.
        votes = np.array([ m.predict(X) for m in self.models ])

        # Classify to 0 or 1 based on majority voting.
        return (np.mean(votes, axis=0) >= 0.5).astype(int)

    # Predict on training data.
    def hill_predict(self, X):
        """
        Predict the class label for training examples that the models
        themselves were trained on. This is used for the hill climbing
        step in ensemble selection.

        Important: X must be the same as that fed into self.fit(X, y)
        during training.

        Input:

            X: ndarray of shape (n_training_examples, # features).
               This is the training design matrix. It must be the same
               ndarray as that passed into self.fit(X, y).

        Output:

            Predicted class labels in the form of an int ndarray
            of shape (n_data_points,). It is either 0 or 1.

        """

        # Input preprocessing.
        X = self.trans.transform(X)

        # Initialize list of predictions. The ith element is a 1-D ndarray
        # that has the predicted outcome of the ith fold.
        predictions = []

        # Predict each fold individually
        fold_size = X.shape[0] / self.n_folds
        for i in range(self.n_folds):

            # For the ith fold, use just the ith model to predict; this was the
            # only model that did not use the fold in training.
            m, X_fold = self.models[i], X[i*fold_size:(i+1)*fold_size]
            predictions.append(m.predict(X_fold))

        # Concatenate the predictions as one ndarray.
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

