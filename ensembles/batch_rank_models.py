#! /nfs/raid13/babar/software/anaconda/bin/python

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_basedir',
                        help=('Directory that contains model_dirs.txt. '))
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
        args.kaggle_root + '/ensembles/' + args.model_basedir + '/model_dirs.txt',
        args.kaggle_root + '/ensembles/' + args.model_basedir + '/model_ranks.txt',
        '--kaggle_root={0}'.format(args.kaggle_root),
    ]
    
    with tempfile.NamedTemporaryFile(mode='w+t', suffix='.submit') as submit_file:
        submit_file.writelines(['universe = vanilla\n',
                                'executable = rank_models_exec.py\n',
                                'getenv = True\n',
                                'arguments = {0}\n'.format(' '.join(arguments)),
                                'output = {0}/rank.out\n'.format(args.model_basedir),
                                'error = {0}/rank.error\n'.format(args.model_basedir),
                                'log = {0}/rank.log\n'.format(args.model_basedir),
                                'accounting_group = group_babar\n',
                                'accounting_group_user = {0}\n'.format(getpass.getuser()),
                                'queue'])
        submit_file.seek(0)
        subprocess.check_call(['condor_submit', submit_file.name])
        time.sleep(1)
        print 'Submitted the following command:'
        print 'rank_models_exec.py {0}'.format(' '.join(arguments))
