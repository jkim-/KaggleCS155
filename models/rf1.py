from pyutils.kaggle_io.extract_inputs import extract_training_data
from input_transforms.trans1 import apply_input_transform
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

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
