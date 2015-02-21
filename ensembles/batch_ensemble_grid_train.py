#! /nfs/raid13/babar/software/anaconda/bin/python

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir',
                        help=('Directory to store the trained ensembles. '
                              'It should also contain model_ranks.txt; '
                              'see rank_models.py. '
                              'This is a relative path w.r.t. this script.' ))
    parser.add_argument('--kaggle_root', default='/nfs/raid13/babar/dchao/KaggleCS155',
                        help='Root directry of the kaggle project.')
    args = parser.parse_args()

    import sys
    sys.path.append(args.kaggle_root)
    import os
    import time
    import itertools
    import tempfile
    import subprocess
    import getpass

    # Grid parameters for ensemble assembly.
    # The baseline is br20_p20_bp20_ni10
    param_list = [
        [ 20 ],                # bag_rounds
        [ 20, 40 ],            # prune_p * 100. (% models to prune. Must be int)
        [ 20, 40, 60 ],        # bag_p * 100.   (% models to bag. Must be int)
        [ 0, 3, 6, 10, 15 ],   # n_init
    ]

    # parameters for debugging purposes
    #param_list = [[ 1 ],[ 100 ],[ 10, 20 ],[ 0, 1 ]]

    # Set directory names
    basedir_abs = args.kaggle_root + '/ensembles/' + args.basedir
    model_ranking_fname = basedir_abs + '/model_ranks.txt'
    ensemble_output_dir = basedir_abs + '/br{0}_p{1}_bp{2}_ni{3}'

    for pars in itertools.product(*param_list):

        # Unpack grid parameters
        bag_rounds, prune_pct, bag_pct, n_init = pars[0], pars[1], pars[2], pars[3]
        curr_ensemble_output_dir = ensemble_output_dir.format(bag_rounds, prune_pct, bag_pct, n_init)

        # Create model directory
        try:
          os.makedirs(curr_ensemble_output_dir + '/model')
          os.chmod(curr_ensemble_output_dir, 0764)
        except OSError:
          pass

        # Assemble arguments for assemble_ensemble_exec.py
        arguments = [
            model_ranking_fname,
            curr_ensemble_output_dir + '/model',
            '--bag_rounds={0}'.format(bag_rounds),
            '--prune_p={0}'.format(prune_pct * 0.01),
            '--bag_p={0}'.format(bag_pct * 0.01),
            '--n_init={0}'.format(n_init),
            '--kaggle_root={0}'.format(args.kaggle_root),
        ]

        # Create temporary submission file, then submit to condor
        with tempfile.NamedTemporaryFile(mode='w+t', suffix='.submit') as submit_file:
            submit_file.writelines(['universe = vanilla\n',
                                    'executable = assemble_ensemble_exec.py\n',
                                    'getenv = True\n',
                                    'arguments = {0}\n'.format(' '.join(arguments)),
                                    'output = {0}/assemble.out\n'.format(curr_ensemble_output_dir),
                                    'error = {0}/assemble.error\n'.format(curr_ensemble_output_dir),
                                    'log = {0}/assemble.log\n'.format(curr_ensemble_output_dir),
                                    'accounting_group = group_babar\n',
                                    'accounting_group_user = {0}\n'.format(getpass.getuser()),
                                    'queue'])
            submit_file.seek(0)
            subprocess.check_call(['condor_submit', submit_file.name])
            time.sleep(1)
            print 'Submitted the following command:'
            print 'assemble_ensemble_exec.py {0}'.format(' '.join(arguments))
