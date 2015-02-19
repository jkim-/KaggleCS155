kaggle_root = '/Users/dchao/Documents/courses/cs155/projects/KaggleCS155'

if __name__ == '__main__':

    import sys
    sys.path.append(kaggle_root)
    from sklearn.externals import joblib
    from pyutils.kaggle_io.extract_inputs import extract_training_data, extract_testing_data
    from pyutils.ensemble_selection.Ensemble import Ensemble
    from scipy.spatial.distance import hamming

    print 'Reading data...'
    Id_train, X_train, y_train = extract_training_data(kaggle_root + '/data/kaggle_train_tf_idf.csv')
    #Id_test, X_test = extract_testing_data('data/kaggle_test_tf_idf.csv')

    print 'Reading models...'
    H = []
    f = open('model_library_ranking.txt', 'r')
    iter_f = iter(f)
    iter_f.next()
    while True:
        try:
            name, err = iter_f.next().split()
            H.append(joblib.load(kaggle_root + '/models/' + name))
        except StopIteration:
            break

    print 'Hill climbing...\n'
    ensemble = Ensemble()
    ensemble.fit(
        X_train, y_train, H,
        n_init=10,
        bag_rounds=20,
        bag_p=0.2,
        prune_p=0.5,
        verbose=True)

    print 'Ensemble hill predict:'
    err = hamming(ensemble.hill_predict(X_train), y_train)
    print err

    #print 'Predicting...'
    #pred = ensemble.predict(X_test)
    #print np.unique(pred)
    #print pred.shape

