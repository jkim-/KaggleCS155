import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from kaggle_io.extract_inputs import extract_training_data

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from CvModel import CvModel

Id, X, y = extract_training_data('data/kaggle_train_tf_idf.csv')

n_folds = 5
scaler = StandardScaler().fit(X)
ada = AdaBoostClassifier()

print 'Training AdaBoost with n_estimators=10'
ada.set_params(n_estimators=10)
cv_ada = CvModel(n_folds, scaler, ada)
cv_ada.fit(X, y)
joblib.dump(cv_ada, 'ada1/1.pkl')

print 'Training AdaBoost with n_estimators=50'
ada.set_params(n_estimators=50)
cv_ada = CvModel(n_folds, scaler, ada)
cv_ada.fit(X, y)
joblib.dump(cv_ada, 'ada1/2.pkl')
