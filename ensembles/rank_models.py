kaggle_root = '../'

import sys
sys.path.append(kaggle_root)
import os
import re
from operator import itemgetter

from pyutils.kaggle_io.extract_inputs import extract_training_data
from pyutils.ensemble_selection.CvModel import CvModel
from sklearn.externals import joblib
from scipy.spatial.distance import hamming

if __name__ == '__main__':

    model_rootdir = kaggle_root + '/models'
    model_dir = [
        'svm_rbf',
        'svm_poly',
        'svm_lin', 
        'rf_std', 
        'rf_bal',
        'bdt_std',

        'bdt_bal',
        'lr_std',
    ]

    prog = re.compile('.*\.pkl$')

    pkl_fnames = []
    for m_dir in model_dir:
        fname_list = os.listdir(model_rootdir + '/' + m_dir)
        for fname in fname_list:
            try:
                pkl_fnames.append(m_dir + '/' + prog.match(fname).group(0))
            except AttributeError:
                continue

    print 'Loading models.'
    models = []
    for p in pkl_fnames:
        print '\t{0}'.format(p)
        models.append((p, joblib.load(model_rootdir + '/' + p)))

    print 'Reading training data.'
    Id, X, y = extract_training_data(kaggle_root + '/data/kaggle_train_tf_idf.csv')

    sys.stdout.write('Scoring')
    sys.stdout.flush()
    hillclimb_errs = []
    for name, m in models:
        sys.stdout.write('.')
        sys.stdout.flush()
        err = hamming(m.hill_predict(X), y)
        hillclimb_errs.append((name, err))
    sys.stdout.write('\n\n')
    sys.stdout.flush()

    f = open('model_library_ranking.txt', 'w')

    print 'Ranking:'
    print

    hillclimb_errs = sorted(hillclimb_errs, key=itemgetter(1))
    title_line = '{0:<20}{1:<20}'.format('model', 'hill_err')
    print title_line
    f.write(title_line + '\n')
    for name, err in hillclimb_errs:
        line = '{0:<20}{1:<20}'.format(name, round(err, 3))
        print line
        f.write(line + '\n')
    print

    f.close()
