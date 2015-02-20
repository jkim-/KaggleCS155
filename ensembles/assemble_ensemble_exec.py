#! /nfs/raid13/babar/software/anaconda/bin/python

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_ranking_fname',
                        help=('File that contains list of models, '
                              'ranked by hillclimb error. '))
    parser.add_argument('ensemble_output_dir',
                        help='Directory to pickle the ensemble.')
    parser.add_argument('--prune_p', type=float, default=0.2,
                        help=('The proportion of models to take. '
                              'e.g. 0.5 means taking the top 20%% '
                              'ranked models in model_ranking_fname for '
                              'ensemble training. '))
    parser.add_argument('--n_init', type=int, default=10,
                        help=('Number of models to initialize the ensemble.'))
    parser.add_argument('--bag_rounds', type=int, default=20,
                        help=('Number of rounds for bagging.'))
    parser.add_argument('--bag_p', type=float, default=0.2,
                        help=('Proportion of models in each bagging round.'))
    parser.add_argument('--rounds', type=int, default=None,
                        help=('Number of rounds for adding to ensemble.'))
    parser.add_argument('--kaggle_root', default='/nfs/raid13/babar/dchao/KaggleCS155',
                        help='Root directry of the kaggle project.')
    parser.add_argument('--predict_fname', default='submit.csv',
                        help=('File to dump prediction output. '
                              'Should also turn on --predict.'))
    parser.add_argument('--hill_predict', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--verbose', action='store_true')
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
    Id_train, X_train, y_train = extract_training_data(args.kaggle_root + '/data/kaggle_train_tf_idf.csv')
    Id_test, X_test = extract_testing_data(args.kaggle_root + '/data/kaggle_test_tf_idf.csv')

    print
    print 'Loading models...'
    model_fnames = []
    with open(args.model_ranking_fname, 'r') as f:
        next(f)
        for line in f:
            model_fnames.append(line.strip().split()[0])
    before_prune_model_count = len(model_fnames)
    model_fnames = model_fnames[:int(args.prune_p*len(model_fnames))]
    after_prune_model_count = len(model_fnames)
    if args.verbose:
        print '\tUsing {0} out of {1} available models.\n'.format(after_prune_model_count,
                                                                  before_prune_model_count)

    H = []
    for name in model_fnames:
        if args.verbose:
            print '\t{0}'.format(name)
        H.append(joblib.load(args.kaggle_root + '/models/' + name))

    print
    print 'Hill climbing...'
    if args.verbose:
        print 'Parameters for hill climbing: '
        print '\t n_init: {0}'.format(args.n_init)
        print '\t bag_rounds: {0}'.format(args.bag_rounds)
        print '\t bag_p: {0}'.format(args.bag_p)
        print '\t rounds: {0}'.format(args.rounds)
        print

    ensemble = Ensemble()
    ensemble.fit(
        X_train, y_train, H,
        n_init=args.n_init,
        bag_rounds=args.bag_rounds,
        bag_p=args.bag_p,
        rounds=args.rounds,
        verbose=args.verbose)

    print
    print 'Pickling trained ensemble... '
    joblib.dump(ensemble, args.ensemble_output_dir + '/ensemble.pkl')

    if args.hill_predict:
        print
        print 'Ensemble hill predicting...'
        err = hamming(ensemble.hill_predict(X_train), y_train)
        print '\tHill climbing error: {0}'.format(err)

    if args.predict:
        print
        print 'Predicting...'
        pred = ensemble.predict(X_test)
        with open(args.predict_fname, 'w') as f:
            f.write('Id,Prediction\n')
            for i, j in zip(Id_test.astype(int).tolist(), pred.tolist()):
                f.write('{0},{1}\n'.format(i, j))
        print '\tOutput written to {0}.'.format(args.predict_fname)

    print
    print 'Done.\n'

    print
