#! /nfs/raid13/babar/software/anaconda/bin/python

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_ranking_fname',
                        help=('File that contains list of models, '
                              'ranked by hillclimb error. '))
    parser.add_argument('ensemble_output_dir',
                        help='Directory to place results.')
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
    parser.add_argument('--hill_predict', action='store_true')
    parser.add_argument('--suppress', action='store_true', 
                        help='Suppress output.')
    args = parser.parse_args()

    import sys
    sys.path.append(args.kaggle_root)
    import os
    import time
    import tempfile
    import subprocess
    import getpass

    try:
      os.makedirs(args.ensemble_output_dir + '/model')
      os.chmod(args.ensemble_output_dir, 0764)
    except OSError:
      pass

    arguments = [
        args.kaggle_root + '/ensembles/' + args.model_ranking_fname,
        args.kaggle_root + '/ensembles/' + args.ensemble_output_dir + '/model',
        '--prune_p={0}'.format(args.prune_p),
        '--n_init={0}'.format(args.n_init),
        '--bag_rounds={0}'.format(args.bag_rounds),
        '--bag_p={0}'.format(args.bag_p),
        '--kaggle_root={0}'.format(args.kaggle_root),
    ]
    if args.rounds: 
        arguments.append('--rounds={0}'.format(args.rounds))
    if args.hill_predict: 
        arguments.append('--hill_predict')
    if not args.suppress: 
        arguments.append('--verbose')
    

    with tempfile.NamedTemporaryFile(mode='w+t', suffix='.submit') as submit_file:
        submit_file.writelines(['universe = vanilla\n',
                                'executable = assemble_ensemble_exec.py\n',
                                'getenv = True\n',
                                'arguments = {0}\n'.format(' '.join(arguments)),
                                'output = {0}/assemble.out\n'.format(args.ensemble_output_dir),
                                'error = {0}/assemble.error\n'.format(args.ensemble_output_dir),
                                'log = {0}/assemble.log\n'.format(args.ensemble_output_dir),
                                'accounting_group = group_babar\n',
                                'accounting_group_user = {0}\n'.format(getpass.getuser()),
                                'queue'])
        submit_file.seek(0)
        subprocess.check_call(['condor_submit', submit_file.name])
        time.sleep(1)
        print 'Submitted the following command:'
        print 'assemble_ensemble_exec.py {0}'.format(' '.join(arguments))
