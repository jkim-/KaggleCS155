# Producing ensembles and their submission files.

## Creating the base directory

You can skip this step if you already have a base directory.

+ Create a base directory to which we save results.

      mkdir {date}_{time}_{version}

  `{date}_{time}_{version}` should uniquely identify the set of base models that we will use. The logic is that the model library is based on all the models that are available in the time indicated.

+ From now on, we call this directory `${BASEDIR}`.

## Create ranked model list.

You can skip this step if you already have a ranked list under `${BASEDIR}` with the name `model_ranks.txt`.

1. Within `${BASEDIR}`, create `model_dirs.txt`. Each line in that file specifies a particular model directory in `../models/`. e.g.:
       rf_std
       bdt_std
       ...

2. Run this command:
       ./batch_rank_models.py `${BASEDIR}`

## Assembling ensembles.

+ Run this command:

      ./batch_ensemble_grid_train.py ${BASEDIR}

## Generating predictions. 

+ Run this command:

      ./batch_ensemble_grid_predict.py ${BASEDIR}

## Collecting submission files.

+ Run this command:

      ./collect_grid_submissions.py ${BASEDIR}

  The submission files will be placed in  `${BASEDIR}/submissions`.

# Final Submission

1. Submit all files under `${BASEDIR}/submissions` to Kaggle. Examine the results. 
2. Use these two files as the final submission:
   + `${BASEDIR}_br20_p20_bp20_ni10.csv`. This is the baseline model; deduced without relying the leaderboard.
   + Among the submitted files (all those under `${BASEDIR}/submissions`), the file that has the best performance on the leaderboard.  This is tuning the "regularization parameters". 
