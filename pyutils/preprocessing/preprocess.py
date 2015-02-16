import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

from numpy import linalg

from sklearn import naive_bayes
from sklearn import metrics
from sklearn import preprocessing
from sklearn import decomposition

train_data_tfidf = pd.read_csv('../data/kaggle_train_tf_idf.csv', index_col='Id')
test_data_tfidf = pd.read_csv('../data/kaggle_test_tf_idf.csv', index_col='Id')

n_data, n_features = train_data_tfidf.shape
train_tfidf = train_data_tfidf.iloc[:n_data/2,:-1]
train_tfidf_label = train_data_tfidf.iloc[:n_data/2,-1]
validation_tfidf = train_data_tfidf.iloc[n_data/2:,:-1]
validation_tfidf_label = train_data_tfidf.iloc[n_data/2:,-1]

scaler = preprocessing.StandardScaler().fit(train_tfidf)
train_tfidf_scaled = scaler.transform(train_tfidf)
max_norm = (linalg.norm(train_tfidf_scaled))**2
reconstruction_error = []
for k in range(0,n_features,5):
    pca = decomposition.PCA(n_components=k, copy=True,
                            whiten=False).fit(train_tfidf_scaled)
    train_tfidf_pca = pca.transform(train_tfidf_scaled)
    reconstruction_error.append([k,
                (max_norm-linalg.norm(train_tfidf_pca)**2)/max_norm])

plt.plot(*zip(*reconstruction_error), lw=3)
plt.axis([0,500,0,1])
plt.xlabel('k')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction error using top-k components')
plt.grid(True)
#plt.savefig('top-k_PCA.png', bbox_inches='tight')
plt.show()
