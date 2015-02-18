import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

from numpy import linalg

from sklearn import naive_bayes
from sklearn import metrics
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import linear_model
from sklearn import grid_search

# Read in data
train_data_tfidf = pd.read_csv('./data/kaggle_train_tf_idf.csv', index_col='Id')
test_data_tfidf = pd.read_csv('./data/kaggle_test_tf_idf.csv', index_col='Id')

# Split data into training and validation sets
n_data, n_features = train_data_tfidf.shape
train_tfidf = train_data_tfidf.iloc[:n_data/2,:-1]
train_tfidf_label = train_data_tfidf.iloc[:n_data/2,-1]
validation_tfidf = train_data_tfidf.iloc[n_data/2:,:-1]
validation_tfidf_label = train_data_tfidf.iloc[n_data/2:,-1]

# Preprocess by centering, scaling, PCA top-200 components, then whiten
scaler = preprocessing.StandardScaler().fit(train_tfidf)
train_tfidf_scaled = scaler.transform(train_tfidf)
validation_tfidf_scaled = scaler.transform(validation_tfidf)

pca = decomposition.PCA(n_components=200, copy=True,
                        whiten=True).fit(train_tfidf_scaled)
train_tfidf_pca = pca.transform(train_tfidf_scaled)
validation_tfidf_pca = pca.transform(validation_tfidf_scaled)

# Compare performance with L1/L2 penalties and various C's using 10-fold CV
param_grid = {'C':[10**x for x in range(-3,3)], 'penalty':['l1', 'l2']}
log_regressor = linear_model.LogisticRegression(fit_intercept=True)
clf = grid_search.GridSearchCV(log_regressor, param_grid, cv=10, scoring='accuracy')
clf.fit(train_tfidf_pca, train_tfidf_label)

print '\n'
print("Best parameters set found on training set:")
print(clf.best_estimator_)

print '\n'
print("Grid scores on training set:")
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
        % (mean_score, scores.std() / 2, params))

print '\n'
print("Score on the validation set:")
y_true, y_pred = validation_tfidf_label, clf.predict(validation_tfidf_pca)
print(metrics.accuracy_score(y_true, y_pred))
