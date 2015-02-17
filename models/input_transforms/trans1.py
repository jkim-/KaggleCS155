import os
from pyutils.kaggle_io.extract_inputs import extract_training_data
from pyutils.preprocessing.InputPreprocessor import InputPreprocessor

# Wrapper to apply the input transformations correctly
def apply_input_transform(X):
    """
    Input:
        X: ndarray. Design matrix of tf_idf data.
    Output:
        X_trans: ndarray. Design matrix of the transformed data. It goes
                 through the following transformations in sequence:

                 1. Center and scale the data.
                 2. Take the top 270 PCA components. The PCA model was
                    deduced using the centered/scaled data from step 1.
    """
    p = InputPreprocessor()

    center_scale_fname = (os.path.dirname(__file__) + '/trans1_center_scale.model')
    X_scaled = p.transform(X, fname=center_scale_fname)

    pca_fname = (os.path.dirname(__file__) + '/trans1_pca.model')
    X_pca = p.transform(X_scaled, fname=pca_fname)

    return X_pca


# Run this as a script to create the transformation models
if __name__ == '__main__':

    # Read data
    Id, X, Y = extract_training_data('data/kaggle_train_tf_idf.csv')

    p = InputPreprocessor()

    # Fit center and scale model
    p.fit_center_scale(X, fname='trans1_center_scale.model')
    X_scaled = p.transform(X, 'center_scale')

    # Fit PCA model
    p.fit_pca(X_scaled,
            n_components=270, whiten=False,
            fname='trans1_pca.model')
    X_pca = p.transform(X_scaled, 'pca')
