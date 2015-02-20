import sys
import csv
import sklearn
import numpy as np

working_path='/nfs/raid13/babar/dchao/KaggleCS155/'
sys.path.append(working_path)

from pyutils.ensemble_selection.CvModel import CvModel
from sklearn.externals import joblib

rankfile_path = 'ensembles/ranks_Feb20_10am.txt'

model_list = []
with open(working_path+rankfile_path, 'rb') as f:
    for row in f:
        modelname, score = row.split()
        model_list.append(modelname)

model = sys.argv[1]
top_models = []
for i, mdl in enumerate(model_list):
    if not len(top_models) < 10: break
    if mdl.startswith(model):
        top_models.append(mdl)
        print 'Loading model', len(top_models), '... (Overall ranking: %d)' % i
        cv_model = joblib.load(working_path+'models/'+mdl)
        print 'Getting its parameters...'
        param_dict = cv_model.models[0].get_params()
        if model == 'bdt_std':
            print 'n_estimators =', param_dict['n_estimators']
            print 'max_depth =', param_dict['base_estimator__max_depth']
        if model == 'bdt_bal':
            print 'n_estimators =', param_dict['n_estimators']
            print 'max_depth =', param_dict['base_estimator__max_depth']
        if model == 'rf_std':
            print 'n_estimators =', param_dict['n_estimators']
            print 'max_features =', param_dict['max_features']
        if model == 'rf_bal':
            print 'n_estimators =', param_dict['n_estimators']
            print 'max_features =', param_dict['max_features']
        if model == 'rf_noip':
            print 'n_estimators =', param_dict['n_estimators']
            print 'max_features =', param_dict['max_features']
        if model == 'rf_lasso':
            print 'n_estimators =', param_dict['n_estimators']
            print 'max_features =', param_dict['max_features']
        if model == 'svm_rbf':
            print 'C =', param_dict['C']
            print 'gamma =', param_dict['gamma']
        if model == 'svm_poly':
            print 'C =', param_dict['C']
            print 'degree =', param_dict['degree']
        if model == 'svm_lin':
            print 'C =', param_dict['C']

