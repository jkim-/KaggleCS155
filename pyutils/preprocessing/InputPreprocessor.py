import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import cPickle as pickle


# Class that manages input preprocessing
class InputPreprocessor:
    """
    This class is a frontend to input preprocessing. It keeps track
    of the list of transformations that can be performed on the input
    data, and allows dumping/loading pickled transform objects.

    Usage example:
        Input: X, an ndarray that is the design matrix.

        # Instantiate a preprocessor
        p = InputPreprocessor()

        # Fit a StandardScaler model from sklearn. Save a pickled
        # model to 'center_scale.model'
        scaler = p.fit_center_scale(X, fname='center_scale.model')

        # Scale X using the fitted center_scale model. The trans_name
        # option specifies that we want to use the cached model.
        X_scaled = p.transform(X, trans_name='center_scale')

        # Scale X using pickled model saved in 'center_scaled.model'.
        X_scaled = p.transform(X, fname='center_scale.model')
    """

    # Keeps an internal dictionary of transforms.
    # (key, value) = (transform_name, transform_object)
    # Note that it only keeps track of the most recently fitted version for
    # each key.
    def __init__(self):
        self.transforms = {}

    # Fit a centering and scaling model.
    def fit_center_scale(self, X, fname=None, **kwargs):
        """
        Fit a model to center and scale the input data. Features are
        standardized by removing the mean and scaling to unit variance.

        Input:

            X: ndarray. The design matrix.

            fname: Pickle the fitted model and save it to fname.
                   Default: None. Nothing is pickled.

            kwargs: Keyword arguments to sklearn.preprocessing.StandardScaler().

        Returns:

            The fitted sklearn.preprocessing.StandardScaler.

        """

        # Fit a StandardScaler and update the internal cache of
        # transformations.
        scaler = StandardScaler(**kwargs).fit(X)
        self.transforms['center_scale'] = scaler

        # Optionally pickle the transform object for later use.
        if fname:
            pickle.dump(scaler, open(fname, 'wb'))

        return scaler

    # Fit PCA.
    def fit_pca(self, X, fname=None, **kwargs):
        """
        Fit a PCA model.

        Input:

            X: ndarray. The design matrix.

            fname: Pickle the fitted model and save it to fname.
                   Default: None. Nothing is pickled.

            kwargs: Keyword arguments to sklearn.decomposition.PCA(). Notable options:

                    n_components: Number of components to keep.

                    whiten: Apply whitening.

        Returns:

            The fitted sklearn.decomposition.PCA object.

        """

        # Fit a sklearn.decomposition.PCA object and update the
        # internal cache of transformations.
        pca = PCA(**kwargs).fit(X)
        self.transforms['pca'] = pca

        # Optionally pickle the transform object for later use.
        if fname:
            pickle.dump(pca, open(fname, 'wb'))
        return pca

    # Transform the data using a fitted model.
    def transform(self, X, trans_name='file', fname=None):
        """
        Transform data using a tranformation model.

        Input:

            X: ndarray. The design matrix to be transformed.

            trans_name: The name of the transformation to apply. This can
                        also be 'file', which will use a pickled object saved
                        in the file specified with the fname parameter.
                        Default: 'file'.

            fname: File name of the pickled transform object.
                   Default: None.

        Returns:

            X_trans: ndarray. This is the transformed version of X.

        """
        trans = None
        if trans_name == 'file':
            if fname:
                trans = pickle.load(open(fname, 'rb'))
        else:
            trans = self.transforms[trans_name]
        return trans.transform(X)


# Unit tests.
if __name__ == '__main__':

    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from kaggle_io.extract_inputs import extract_training_data

    print
    print 'Reading in data/kaggle_train_tf_idf.csv'
    Id, X, Y = extract_training_data('data/kaggle_train_tf_idf.csv')
    print 'First 10 features of first record:'
    print X[0,0:10]
    print

    p = InputPreprocessor()

    print '1. Input center/scale test.\n'
    p.fit_center_scale(X)
    X_scaled = p.transform(X, 'center_scale')
    print 'With cached model:'
    print X_scaled[0,0:10]
    p.fit_center_scale(X, fname='center_scale.model')
    X_scaled = p.transform(X, fname='center_scale.model')
    print 'With pickled model:'
    print X_scaled[0,0:10]
    print

    print '2. PCA test. With parameters n_components=5, whiten=False.'
    p.fit_pca(X_scaled, n_components=5, copy=True, whiten=False)
    X_pca = p.transform(X_scaled, 'pca')
    print 'With cached model:'
    print X_pca[0,0:5]
    p.fit_pca(X_scaled, n_components=5, copy=True, whiten=False, fname='pca.model')
    X_pca = p.transform(X_scaled, fname='pca.model')
    print 'With pickled model:'
    print X_pca[0,0:5]
    print
