import numpy as np
from sklearn.externals import joblib
from scipy.spatial.distance import hamming

# Class for an ensemble of cross-validated models.
class Ensemble:

    def __init__(self):
        self.models = []

    # This is an internal method used for hill climbing.
    def hill_predict(self, models, X):
        """
        Predict in sample labels. Internal method used for hill-climbing.

        Input:

            X: ndarray of training examples. This must be the same
               data used when training models.

            models: List of CvModels trained using X.

        Output:

            int ndarray of in sample predictions on X.

        """

        # Average the in-sample predictions from each model and take a majority
        # vote.
        y_pred = np.zeros(X.shape[0], dtype=float)
        for m in models:
            y_pred += m.hill_predict(X)
        y_pred /= len(models)
        return (y_pred >= 0.5).astype(int)

    # Fit an ensemble of models.
    def fit(self, X, y, H, rounds=None):
        """
        This fits an ensemble of models given a library of models. This method
        implements what is known as "ensemble selection".

        Input:

            H: List of CvModels. This is the library of models to choose from.

            X: ndarray of training examples. This must be the same data used
               to train each model in H.

            y: ndarray of training example lables.

            rounds: Number of rounds for model selection. Default: len(H).

        Output:
            Populates self.models with models from H.
            Returns nothing.
        """

        # Initialize ensemble
        self.models = []

        # Foward selection
        if rounds is None: rounds = len(H)
        for i in range(rounds):

            # Find the model that improves the overall ensemble performance the
            # most. This is measured by the hill climbing cross validation error.
            err_hmin, hmin = 1.0, None
            for h in H:
                y_pred = self.hill_predict(self.models + [ h ], X)
                err = hamming(y_pred, y.astype(int))
                if err < err_hmin:
                    err_hmin, hmin = err, h
            self.models.append(hmin)

    # Use the ensemble to predict labels for unseen/test examples.
    def predict(self, X):
        """
        Predict out of sample labels.

        Input:

            X: ndarray of unseen/test examples.

        Output:

            int ndarray of predictions on X.

        """
        # Average the in-sample predictions from each model in self.models
        # and take a majority vote.
        y_pred = np.zeros(X.shape[0], dtype=float)
        for m in self.models:
            y_pred += m.predict(X)
        y_pred /= len(self.models)
        return (y_pred >= 0.5).astype(int)

if __name__ == '__main__':

    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from kaggle_io.extract_inputs import extract_training_data, extract_testing_data

    print 'Reading data...'
    Id_train, X_train, y_train = extract_training_data('data/kaggle_train_tf_idf.csv')
    Id_test, X_test = extract_testing_data('data/kaggle_test_tf_idf.csv')

    print 'Reading models...'
    H = []
    H.append(joblib.load('rf1/1.pkl'))
    H.append(joblib.load('rf1/2.pkl'))
    H.append(joblib.load('rf1/3.pkl'))
    H.append(joblib.load('ada1/1.pkl'))
    H.append(joblib.load('ada1/2.pkl'))

    print 'Hill climbing...'
    ensemble = Ensemble()
    ensemble.fit(X_train, y_train, H)

    print 'Predicting...'
    pred = ensemble.predict(X_test)
    print np.unique(pred)
    print pred.shape

