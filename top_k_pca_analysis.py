import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

from pyutils.preprocessing.InputPreprocessor import InputPreprocessor
from pyutils.kaggle_io.extract_inputs import extract_training_data

# Read in tf_idf training data.
Id, X, Y = extract_training_data('data/kaggle_train_tf_idf.csv')

# Get an input preprocessor
p = InputPreprocessor()

# Center and scale the data
p.fit_center_scale(X)
X_scaled = p.transform(X, 'center_scale')

# Compute the square norm of keeping only the top k PCA components
k_list, sq_norm_list = range(1, X_scaled.shape[1], 5), []
for k in k_list:
    p.fit_pca(X_scaled, n_components=k)
    X_pca = p.transform(X_scaled, 'pca')
    sq_norm_list.append(linalg.norm(X_pca, 'fro')**2)
k_arr, sq_norm_arr = np.array(k_list), np.array(sq_norm_list)

# Compute the reconstruction error
max_sq_norm = (linalg.norm(X_scaled, 'fro'))**2
reconstruction_error_arr = 1 - sq_norm_arr / max_sq_norm

# Plot and save figure
fig = plt.figure(figsize=(1.618 * 6, 6))
fig.patch.set_facecolor('white')
ax = fig.add_subplot(1, 1, 1)
for l in ax.get_xticklabels(): l.update({'fontsize':16})
for l in ax.get_yticklabels(): l.update({'fontsize':16})
plt.plot(k_arr, reconstruction_error_arr, 'b-', lw=3)
plt.title(r'Reconstruction error v.s. top-$k$ components', fontsize=20)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel('Reconstruction Error', fontsize=20)
plt.grid(True)
plt.savefig('plots/top-k_PCA.pdf', bbox_inches='tight')
plt.show()

