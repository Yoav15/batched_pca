import numpy as np
from scipy.linalg import eig
import time


class MyPCA:

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        # Standardize data
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        X_std = (X - self.mean) / self.scale

        # Eigendecomposition of covariance matrix - which is symmetric
        cov_mat = np.cov(X_std.T)
        eig_vals, eig_vecs = np.linalg.eigh(cov_mat)

        # Adjusting the eigenvectors that are largest in absolute value to be positive
        max_abs_idx = np.argmax(np.abs(eig_vecs), axis=0)
        signs = np.sign(eig_vecs[max_abs_idx, range(eig_vecs.shape[0])])
        eig_vecs = eig_vecs * signs[np.newaxis, :]
        eig_vecs = eig_vecs.T

        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[i, :]) for i in range(len(eig_vals))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        eig_vals_sorted = np.array([x[0] for x in eig_pairs])
        eig_vecs_sorted = np.array([x[1] for x in eig_pairs])

        self.components = eig_vecs_sorted[:self.n_components, :]

        # Explained variance ratio
        self.explained_variance_ratio = [i / np.sum(eig_vals) for i in eig_vals_sorted[:self.n_components]]

        self.cum_explained_variance = np.cumsum(self.explained_variance_ratio)

        return self

    def transform(self, X):
        X = X.copy()
        X_std = (X - self.mean) / self.scale
        X_proj = X_std.dot(self.components.T)

        return X_proj


class MyBatchedPCA:

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        # Standardize data
        t0 = time.perf_counter()
        self.mean = np.mean(X, axis=1, keepdims=True)
        self.scale = np.std(X, axis=1, keepdims=True)
        X_std = (X - self.mean) / self.scale
        t1 = time.perf_counter()
        #print(f'a : {t1 - t0}')

        # Eigendecomposition of covariance matrix
        # np.cov(X_std.T)

        #########################
        # cov
        t0 = time.perf_counter()
        # Note that we should center the data before doing covaraince but it is already centered
        # X_std -= X_std.mean(axis=1, keepdims=True)
        fact = X_std.shape[1] - 1
        cov_mat = np.matmul(X_std.transpose(0, 2, 1), X_std) / fact
        t1 = time.perf_counter()
        #print(f'b : {t1 - t0}')
        #########################
        t0 = time.perf_counter()
        # eigh for symmetric matrix
        eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
        t1 = time.perf_counter()
        #print(f'c : {t1 - t0}')

        # Adjusting the eigenvectors that are largest in absolute value to be positive
        t0 = time.perf_counter()
        max_abs_idx = np.argmax(np.abs(eig_vecs), axis=1)
        
        signs = np.sign(eig_vecs[
            np.arange(max_abs_idx.shape[0])[:, None],
            np.arange(max_abs_idx.shape[1]),
            max_abs_idx
            ])
        
        eig_vecs = eig_vecs * signs[:, np.newaxis, :]
        eig_vecs = eig_vecs.transpose(0, 2, 1)
        
        abs_eig_vals = np.abs(eig_vals)
        eig_pairs = np.concatenate((abs_eig_vals[:, :, np.newaxis], eig_vecs), axis=-1)
        t1 = time.perf_counter()
        #print(f'd : {t1 - t0}')
        
        t0 = time.perf_counter()
        my_key = eig_pairs[:, :, 0]
        inds = np.flip(np.lexsort(keys = [my_key], axis=1), axis=1)
        eig_pairs = eig_pairs[np.arange(eig_pairs.shape[0])[:, None], inds]
        t1 = time.perf_counter()
        #print(f'e : {t1 - t0}')

        t0 = time.perf_counter()
        eig_vals_sorted = eig_pairs[:, :, 0] # np.array([[x[0] for x in sample] for sample in eig_pairs])
        eig_vecs_sorted = eig_pairs[:, :, 1:] # np.array([[x[1] for x in sample] for sample in eig_pairs])

        self.components = eig_vecs_sorted[:, :self.n_components, :]

        # Explained variance ratio
        self.explained_variance_ratio = eig_vals_sorted[:, :self.n_components] / np.sum(eig_vals_sorted, axis=1, keepdims=True)
        # [[i / np.sum(sample) for i in sample[:self.n_components]] for sample in eig_vals_sorted]

        self.cum_explained_variance = np.cumsum(self.explained_variance_ratio, axis=1)
        t1 = time.perf_counter()
        #print(f'f : {t1 - t0}')

        return self

    def transform(self, X):
        X = X.copy()
        X_std = (X - self.mean) / self.scale
        X_proj = X_std.dot(self.components.T)

        return X_proj