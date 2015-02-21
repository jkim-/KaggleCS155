#! /nfs/raid13/babar/software/anaconda/bin/python

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ensemble_output_dir',
                        help=('Directory to place results. This directory must '
                              'contain model/ensemble.pkl that stores the pickled '
                              'ensemble.'))
    parser.add_argument('--hill_predict', action='store_true')
    parser.add_argument('--kaggle_root', default='/nfs/raid13/babar/dchao/KaggleCS155',
                        help='Root directry of the kaggle project.')
    args = parser.parse_args()

    import sys
    sys.path.append(args.kaggle_root)
    import os
    import time
    import tempfile
    import subprocess
    import getpass

    arguments = [
        args.kaggle_root + '/ensembles/' + args.ensemble_output_dir + '/model',
        args.kaggle_root + '/ensembles/' + args.ensemble_output_dir + '/submit.csv',
        '--kaggle_root={0}'.format(args.kaggle_root),
    ]
    if args.hill_predict: 
        arguments.append('--hill_predict')
    

    with tempfile.NamedTemporaryFile(mode='w+t', suffix='.submit') as submit_file:
        submit_file.writelines(['universe = vanilla\n',
                                'executable = predict_submission_exec.py\n',
                                'getenv = True\n',
                                'arguments = {0}\n'.format(' '.join(arguments)),
                                'output = {0}/predict.out\n'.format(args.ensemble_output_dir),
                                'error = {0}/predict.error\n'.format(args.ensemble_output_dir),
                                'log = {0}/predict.log\n'.format(args.ensemble_output_dir),
                                'accounting_group = group_babar\n',
                                'accounting_group_user = {0}\n'.format(getpass.getuser()),
                                'queue'])
        submit_file.seek(0)
        subprocess.check_call(['condor_submit', submit_file.name])
        time.sleep(1)
        print 'Submitted the following command:'
        print 'predict_submission_exec.py {0}'.format(' '.join(arguments))
