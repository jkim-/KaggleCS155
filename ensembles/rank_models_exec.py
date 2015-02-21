#! /nfs/raid13/babar/software/anaconda/bin/python

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dirlist_fname',
                        help=('File that contains list of model directories. '
                              'Each line is a directory; do not include any prefixes. '
                              'i.e. bdt1 instead of args.kaggle_root/models/bdt1'))
    parser.add_argument('output_ranking_fname',
                        help='Ranking output file. Will overwrite.')
    parser.add_argument('--kaggle_root', default='/nfs/raid13/babar/dchao/KaggleCS155',
                        help='Root directry of the kaggle project.')
    args = parser.parse_args()

    import sys
    sys.path.append(args.kaggle_root)
    import os
    import re
    from operator import itemgetter

    from pyutils.kaggle_io.extract_inputs import extract_training_data
    from pyutils.ensemble_selection.CvModel import CvModel
    from sklearn.externals import joblib
    from scipy.spatial.distance import hamming

    model_dirs = []
    with open(args.model_dirlist_fname, 'r') as f:
        for line in f:
            model_dirs.append(line.strip())

    print 'Reading training data.\n'
    Id, X, y = extract_training_data(args.kaggle_root + '/data/kaggle_train_tf_idf.csv')

    print 'Scoring models.'
    hillclimb_errs = []
    prog, model_rootdir = re.compile('.*\.pkl$'), args.kaggle_root + '/models'
    for m_dir in model_dirs:
        dir_contents = os.listdir(model_rootdir + '/' + m_dir)
        for fname in dir_contents:
            try:
                model_name = m_dir + '/' + prog.match(fname).group(0)
                print '\t{0}'.format(model_name)
                model = joblib.load(model_rootdir + '/' + model_name)
                err = hamming(model.hill_predict(X), y)
                hillclimb_errs.append((model_name, err))

            except AttributeError:
                continue

    print '\nRanking models.'

    with open(args.output_ranking_fname, 'w') as f:

        header_line = '{0:<20}{1:<20}'.format('model', 'hill_err')
        f.write(header_line + '\n')

        hillclimb_errs = sorted(hillclimb_errs, key=itemgetter(1))
        for name, err in hillclimb_errs:
            line = '{0:<20}{1:<20}'.format(name, round(err, 3))
            f.write(line + '\n')

    print '\tOutput written to {0}.'.format(args.output_ranking_fname)
    print
