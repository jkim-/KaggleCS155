from pyutils.kaggle_io.extract_inputs import extract_training_data
from input_transforms.trans1 import apply_input_transform
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

def get_folds(n_folds, Id, X, Y):
    pts_per_fold = Id.shape[0] / n_folds

    folds = []
    for i in range(n_folds):
        pass

print 'Reading and transforming data...'
Id, X, Y = extract_training_data('data/kaggle_train_tf_idf.csv')
X_trans = apply_input_transform(X)
print

print 'Training random forest...'
rf = RandomForestClassifier(
    n_estimators=100,
    max_features='auto'
)
rf.fit(X, Y)
print

joblib.dump(rf, 'random_forests/rf1.model')
