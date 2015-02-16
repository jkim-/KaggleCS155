import numpy as np

# Use genfromtxt for file import
genfromtxt = None
try:
    import iopro
    genfromtxt= iopro.genfromtxt
except ImportError:
    genfromtxt= np.genfromtxt

# Extract ndarrays from KaggleChallenge1 training files.
def extract_training_data(fname):
    """
    Input:

        fname: The path to the training file. It can be either the
               wc or tf_idf file; e.g. 'kaggle_train_{wc|tf_idf}.csv'.

    Output:

        Id: ndarray of record Id's.
            This corresponds to column 0 in the file.

        X: ndarray for the design matrix; i.e. each row is a record,
           and each column is a feature. These correspond to
           columns 1 - second to last column in the file.

        Y: ndarray of the target label.
           This corresponds to the last column in the file.
    """

    # Read file. Skip the header line.
    arr = genfromtxt(fname, delimiter=',', skip_header=1)

    # Column order: Id, feature1, ..., featureD, Label.
    Id, X, Y = arr[:, 1], arr[:,1:-1], arr[:,-1]

    return Id, X, Y


# Extract ndarrays from KaggleChallenge1 testing files.
def extract_testing_data(fname):
    """
    Input:

        fname: The path to the testing file. It can be either the
               wc or tf_idf file; e.g. 'kaggle_test_{wc|tf_idf}.csv'.

    Output:

        Id: ndarray of record Id's.
            This corresponds to column 0 in the file.

        X: ndarray for the design matrix; i.e. each row is a record,
           and each column is a feature. These correspond to
           columns 1 - second to last column in the file.

        Y: ndarray of the target label.
           This corresponds to the last column in the file.
    """

    # Read file. Skip the header line.
    arr = genfromtxt(fname, delimiter=',', skip_header=1)

    # Column order: Id, feature1, ..., featureD
    Id, X = arr[:, 1], arr[:,1:]

    return Id, X


# Unit tests
if __name__ == '__main__':

    print 'Extracting kaggle_train_tf_idf.csv:'
    Id, X, Y = extract_training_data('data/kaggle_train_tf_idf.csv')
    print 'Id: ', Id.shape
    print 'X: ', X.shape
    print 'Y: ', Y.shape
    print


    print 'Extracting kaggle_train_wc.csv:'
    Id, X, Y = extract_training_data('data/kaggle_train_wc.csv')
    print 'Id: ', Id.shape
    print 'X: ', X.shape
    print 'Y: ', Y.shape
    print

    print 'Extracting kaggle_test_tf_idf.csv:'
    Id, X = extract_testing_data('data/kaggle_test_tf_idf.csv')
    print 'Id: ', Id.shape
    print 'X: ', X.shape
    print


    print 'Extracting kaggle_test_wc.csv:'
    Id, X = extract_testing_data('data/kaggle_test_wc.csv')
    print 'Id: ', Id.shape
    print 'X: ', X.shape
    print
