import sys
import csv
import sklearn
import numpy as np

working_path='/nfs/raid13/babar/dchao/KaggleCS155/'
sys.path.append(working_path)

from pyutils.ensemble_selection.CvModel import CvModel
from sklearn.externals import joblib

rankfile_path = 'ensembles/Feb20_6pm/model_ranks.txt'

model_list = []
errors = []
with open(working_path+rankfile_path, 'rb') as f:
    next(f)
    for row in f:
        modelname, error = row.split()
        model_list.append(modelname)
        errors.append(error)

clf_list = []
for mdl in model_list:
    clf = mdl.split('/')[0]
    clf_list.append(clf)
print '\n'
print 'This ranking includes:\n', sorted(list(set(clf_list)))

model = sys.argv[1]
print '\n'
print '''Ranking top 10 {0}'s out of {1} models:'''.format(sys.argv[1], len(model_list))
top_models = []
print '\n'
for i, mdl in enumerate(model_list):
    if not len(top_models) < 10: break
    if mdl.split('/')[0] == model:
        top_models.append(mdl)
        print 'Loading model', len(top_models), '(Overall ranking: {0} with error {1})...'.format(i, errors[i])
        cv_model = joblib.load(working_path+'models/'+mdl)
        print 'Getting its parameters...'
        param_dict = cv_model.models[0].get_params()
        param_dict_trans = cv_model.trans.get_params()
        if model == 'bdt_std' or model == 'bdt_bal':
            print 'n_estimators =', param_dict['n_estimators']
            print 'max_depth =', param_dict['base_estimator__max_depth']
        if model.startswith('rf_'):
            print 'n_estimators =', param_dict['n_estimators']
            print 'max_features =', param_dict['max_features']
        if model == 'svm_rbf':
            print 'C =', param_dict['C']
            print 'gamma =', param_dict['gamma']
            print 'n_components =', param_dict_trans['pca__n_components']
            print 'whiten =', str(param_dict_trans['pca__whiten'])
        if model == 'svm_poly':
            print 'C =', param_dict['C']
            print 'degree =', param_dict['degree']
        if model == 'svm_lin':
            print 'C =', param_dict['C']
        if model == 'lr_std' or model == 'lr_bal':
            print 'penalty =', param_dict['penalty']
            print 'C =', param_dict['C']
            print 'fit_intercept =', str(param_dict['fit_intercept'])
            print 'n_components =', param_dict_trans['pca__n_components']
        if model == 'gtb_std':
            print 'loss =', param_dict['loss']
            print 'learning_rate =', param_dict['learning_rate']
            print 'n_estimators =', param_dict['n_estimators']
            print 'max_depth =', param_dict['max_depth']
            print 'subsample =', param_dict['subsample']
        print '-'*80
