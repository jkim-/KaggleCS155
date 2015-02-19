import sys
import numpy as np
from sklearn.externals import joblib
from scipy.spatial.distance import hamming

# Class for an ensemble of cross-validated models.
class Ensemble:

    def __init__(self):
        self.models = []

    # This is an internal method used for hill climbing.
    def hill_predict(self, X):
        """
        Predict in sample labels.

        Input:

            X: ndarray of training examples. This must be the same
               data used when training models.

        Output:

            int ndarray of in sample predictions on X.

        """

        # Average the in-sample predictions from each model and take a majority
        # vote.
        y_pred = np.zeros(X.shape[0], dtype=float)
        for m in self.models:
            y_pred += m.hill_predict(X)
        y_pred /= len(self.models)
        return (y_pred >= 0.5).astype(int)

    # Fit an ensemble of models.
    def fit(self, X, y, H,
            n_init=0,
            rounds=None,
            verbose=False):
        """
        This fits an ensemble of models given a library of models. This method
        implements what is known as "ensemble selection".

        Input:

            H: List of CvModels. This is the library of models to choose from.

            X: ndarray of training examples. This must be the same data used
               to train each model in H.

            y: ndarray of training example lables.

            n_init: Number of models to initialize the ensemble. This takes the
                    first n_init models from H.
                    Default: 0.

            rounds: Number of rounds for model selection. Default: len(H).

            verbose: Print diagnostics. Default: False.

        Output:
            Populates self.models with models from H.
            Returns nothing.
        """

        if rounds is None:
            rounds = len(H)

        # Initialize ensemble
        self.models = H[:n_init]
        votes = np.zeros(X.shape[0], dtype=float)
        if verbose:
            print 'Initial ensemble size: {0}'.format(len(self.models))

        # Foward selection
        for i in range(rounds):

            err_min, hmin_pred, hmin = 1.0, None, None
            for h in H:
                h_pred = h.hill_predict(X)
                enh_pred = ((votes + h_pred) / (i+1) >= 0.5).astype(int)
                err = hamming(enh_pred, y.astype(int))
                if err < err_min:
                    err_min, hmin_pred, hmin = err, h_pred, h

            self.models.append(hmin)
            votes += hmin_pred

            if verbose:
                print 'Hillclimb round: {0}, Hillclimb error: {1}'.format(i+1, err_min)

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
    ensemble.fit(X_train, y_train, H, n_init=2, verbose=True)

    print 'Predicting in sample...'
    pred = ensemble.hill_predict(X_train)
    print np.unique(pred)
    print pred.shape


    print 'Predicting out of sample...'
    pred = ensemble.predict(X_test)
    print np.unique(pred)
    print pred.shape

