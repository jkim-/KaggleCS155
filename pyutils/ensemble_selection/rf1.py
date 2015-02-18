import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from kaggle_io.extract_inputs import extract_training_data

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from CvModel import CvModel

Id, X, y = extract_training_data('data/kaggle_train_tf_idf.csv')

n_folds = 5
scaler = StandardScaler().fit(X)
rf = RandomForestClassifier()

print 'Training RandomForest with n_estimators=10'
rf.set_params(n_estimators=10)
cv_rf = CvModel(n_folds, scaler, rf)
cv_rf.fit(X, y)
joblib.dump(cv_rf, 'rf1/1.pkl')

print 'Training RandomForest with n_estimators=50'
rf.set_params(n_estimators=50)
cv_rf = CvModel(n_folds, scaler, rf)
cv_rf.fit(X, y)
joblib.dump(cv_rf, 'rf1/2.pkl')

print 'Training RandomForest with n_estimators=100'
rf.set_params(n_estimators=100)
cv_rf = CvModel(n_folds, scaler, rf)
cv_rf.fit(X, y)
joblib.dump(cv_rf, 'rf1/3.pkl')
