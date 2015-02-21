#! /nfs/raid13/babar/software/anaconda/bin/python

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir',
                        help=('Directory where the ensembles are stored. '
                              'The submissions directory will be created here. '
                              'Each ensemble\'s submission is under '
                              'basedir/br[0-9]+_p[0-9]+_bp[0-9]+_ni[0-9]+/submit.csv '))
    parser.add_argument('--kaggle_root', default='/nfs/raid13/babar/dchao/KaggleCS155',
                        help='Root directry of the kaggle project.')
    args = parser.parse_args()

    import sys
    sys.path.append(args.kaggle_root)
    import os
    import re
    import shutil

    # Get the list of ensemble base directories
    basedir_abs = args.kaggle_root + '/ensembles/' + args.basedir
    ensemble_basedirs = []
    for fname in os.listdir(basedir_abs):
        try:
            match = re.match('br[0-9]+_p[0-9]+_bp[0-9]+_ni[0-9]+', fname)
            ensemble_basedirs.append(match.group(0))
        except AttributeError:
            continue

    # Create submissions directory
    submission_dir = basedir_abs + '/submissions'
    try:
      os.makedirs(submission_dir)
      os.chmod(submission_dir, 0764)
    except OSError:
      pass

    # Collect submission files into a single directory
    for ensemble_dir in ensemble_basedirs:

        ensemble_submit_fname = basedir_abs + '/' + ensemble_dir + '/submit.csv'
        submission_fname = submission_dir + '/' + args.basedir + '_' + ensemble_dir + '.csv'
        try:
            shutil.copyfile(ensemble_submit_fname, submission_fname)
        except IOErroor:
            print 'Error when copying {0} to {1}'.format(ensemble_submit_fname, submission_fname)
