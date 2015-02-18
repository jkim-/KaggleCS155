#!/nfs/raid13/babar/software/anaconda/bin/python
import sys
import numpy as np
import sklearn as sk
from datetime import datetime

from models.CvModel import CvModel
from pyutils.kaggle_io.extract_inputs import extract_training_data

from sklearn import decomposition
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import pipeline
from sklearn import cross_validation
from sklearn import externals

# n_estimators, max_features, max_depth, min_samples_leaf, bootstrap, model_index = sys.argv
clf_params = sys.argv[1:]
for i in range(len(clf_params)):
    if clf_params[i] == 'None':
        clf_params[i] = None
        #print(clf_params[i], ' became None.')
    elif (clf_params[i] == 'True' or clf_params[i] == 'False'):
        clf_params[i] = bool(clf_params[i])
        #print(clf_params[i], ' became a boolean.')
    else:
        try:
            clf_params[i] = int(clf_params[i])
        except ValueError:
            clf_params[i] = float(clf_params[i])
        #print(clf_params[i], 'became an integer.')

# Start stopwatch
start_time = datetime.now()

# Read in tf_idf training data.
Id, X, Y = extract_training_data('data/kaggle_train_tf_idf.csv')

n_folds = 5
scaler = preprocessing.StandardScaler()
pca = decomposition.PCA(n_components=200, whiten=True)
preprocess_pipe = pipeline.Pipeline([('scale_center',scaler),('pca',pca)]).fit(X)
rf_model = ensemble.RandomForestClassifier(n_estimators=clf_params[0],
                                           max_features=clf_params[1],
                                           max_depth=clf_params[2],
                                           min_samples_leaf=clf_params[3],
                                           bootstrap=clf_params[4])
cv_rf = CvModel(n_folds, preprocess_pipe, rf_model)
print 'Fitting the data...'
cv_rf.fit(X, Y)
print 'Pickling the models...'
externals.joblib.dump(cv_rf, '/home/jkim/Kaggle/CS155/models/random_forests/test/test_rf1_'+str(clf_params[5])+'.pkl')
#cv_rf_pkl = joblib.load('models/random_forests/rf1_1.pkl')
#print np.unique(cv_rf_pkl.hill_predict(X_train) == cv_rf.hill_predict(X_train))
print 'This took', datetime.now() - start_time, 'seconds.'
