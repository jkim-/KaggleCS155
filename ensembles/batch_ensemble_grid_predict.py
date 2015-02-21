#! /nfs/raid13/babar/software/anaconda/bin/python

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir',
                        help=('Directory where the ensembles are stored. '
                              'Each ensemble is pickled under '
                              'basedir/br[0-9]+_p[0-9]+_bp[0-9]+_ni[0-9]+/model'))
    parser.add_argument('--kaggle_root', default='/nfs/raid13/babar/dchao/KaggleCS155',
                        help='Root directry of the kaggle project.')
    args = parser.parse_args()

    import sys
    sys.path.append(args.kaggle_root)
    import os
    import re
    import time
    import itertools
    import tempfile
    import subprocess
    import getpass

    # Get the list of ensemble base directories
    basedir_abs = args.kaggle_root + '/ensembles/' + args.basedir
    ensemble_basedirs = []
    for fname in os.listdir(basedir_abs):
        try:
            match = re.match('br[0-9]+_p[0-9]+_bp[0-9]+_ni[0-9]+', fname)
            ensemble_basedirs.append(match.group(0))
        except AttributeError:
            continue

    # Submit jobs for each pickled ensemble 
    for ensemble_dir in ensemble_basedirs:

        # Arguments for predict_submission_exec.py
        ensemble_dir_abs = basedir_abs + '/' + ensemble_dir
        arguments = [
            ensemble_dir_abs + '/model',
            ensemble_dir_abs + '/submit.csv',
            '--kaggle_root={0}'.format(args.kaggle_root),
            '--hill_predict',
        ]

        # Create temporary submission file, then submit to condor
        with tempfile.NamedTemporaryFile(mode='w+t', suffix='.submit') as submit_file:
            submit_file.writelines(['universe = vanilla\n',
                                    'executable = predict_submission_exec.py\n',
                                    'getenv = True\n',
                                    'arguments = {0}\n'.format(' '.join(arguments)),
                                    'output = {0}/predict.out\n'.format(ensemble_dir_abs),
                                    'error = {0}/predict.error\n'.format(ensemble_dir_abs),
                                    'log = {0}/predict.log\n'.format(ensemble_dir_abs),
                                    'accounting_group = group_babar\n',
                                    'accounting_group_user = {0}\n'.format(getpass.getuser()),
                                    'queue'])
            submit_file.seek(0)
            subprocess.check_call(['condor_submit', submit_file.name])
            time.sleep(1)
            print 'Submitted the following command:'
            print 'predict_submission_exec.py {0}'.format(' '.join(arguments))
