if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ensemble_dir',
                        help='Directory to where the ensemble is stored.')
    parser.add_argument('submission_fname',
                        help=('File to save prediction output. '))
    parser.add_argument('--hill_predict', action='store_true')
    parser.add_argument('--kaggle_root', default='../',
                        help='Root directry of the kaggle project.')
    args = parser.parse_args()

    import sys
    sys.path.append(args.kaggle_root)
    import numpy as np
    from sklearn.externals import joblib
    from pyutils.kaggle_io.extract_inputs import extract_training_data, extract_testing_data
    from pyutils.ensemble_selection.Ensemble import Ensemble
    from scipy.spatial.distance import hamming

    print
    print 'Reading data...'
    Id_test, X_test = extract_testing_data(args.kaggle_root + '/data/kaggle_test_tf_idf.csv')

    print
    print 'Loading ensemble...'
    ensemble = joblib.load(args.ensemble_dir + '/ensemble.pkl')

    if args.hill_predict:
        print
        print 'Ensemble hill predicting...'
        Id_train, X_train, y_train = extract_training_data(args.kaggle_root + '/data/kaggle_train_tf_idf.csv')
        err = hamming(ensemble.hill_predict(X_train), y_train)
        print '\tHill climbing error: {0}'.format(err)

    print
    print 'Predicting...'
    pred = ensemble.predict(X_test)
    with open(args.submission_fname, 'w') as f:
        f.write('Id,Prediction\n')
        for i, j in zip(Id_test.astype(int).tolist(), pred.tolist()):
            f.write('{0},{1}\n'.format(i, j))
    print '\tOutput written to {0}.'.format(args.submission_fname)

    print
    print 'Done.\n'

    print
