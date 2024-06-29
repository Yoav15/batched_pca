import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from pcas import MyPCA, MyBatchedPCA
from itertools import product
import matplotlib.pyplot as plt

def get_data(datasets=50, samples=1000, features=100):
  X = np.concatenate([make_blobs(n_samples=samples, n_features=features)[0][np.newaxis, ...] for _ in range(datasets)])
  return X

def my_batched_pca(X, N, comp=2):
  # ---------------------------------------------------------
  t0 = time.perf_counter()
  for _ in range(N):
    my_pca = MyBatchedPCA(n_components = comp).fit(X)
  t1 = time.perf_counter()
  
  #print('Components:\n', my_pca.components[-1, :, :2])
  # #print('Explained variance ratio from scratch:\n', my_pca.explained_variance_ratio)
  return t1 - t0


def my_pca(X, N, comp=2):
  # ---------------------------------------------------------
  t0 = time.perf_counter()
  for _ in range(N):
    for dataset in X:
      my_pca = MyPCA(n_components = comp).fit(dataset)
  t1 = time.perf_counter()

  #print('Components:\n', my_pca.components[:, :2])
  # #print('Explained variance ratio from scratch:\n', my_pca.explained_variance_ratio)
  return t1 - t0


def np_pca(X, N, comp=2):
  t0 = time.perf_counter()
  for _ in range(N):
    for dataset in X:
      pca = PCA(n_components = comp, svd_solver='covariance_eigh').fit(StandardScaler().fit_transform(dataset))
  t1 = time.perf_counter()

  #print('Components:\n', pca.components_[:, :2])
  # #print('Explained variance ratio:\n', pca.explained_variance_ratio_)
  return t1 - t0


def grid_search():
  N = 1
  datasets = np.array([150])
  samples = np.array([200])
  features = np.array([10])

  grid_results = np.zeros((len(datasets), len(samples), len(features), 3))
  grid = np.zeros((len(datasets), len(samples), len(features), 3))

  for a, b, c in product(datasets, samples, features):
    #print(a, b, c)
    X = get_data(a, b, c)
    N = 1
    t_batched = my_batched_pca()
    t_reg = my_pca()
    t_sk = np_pca()
    loc = np.where(datasets == a)[0], np.where(samples == b)[0], np.where(features == c)[0] 
    grid_results[loc] = [t_batched, t_reg, t_sk]
    grid[loc] = [a, b, c]
    
  #print(grid_results)
  # #print(grid)
  
# grid_search()

def feature_count_analysis():
  N_repeat = 1
  datasets = 150
  samples = 200
  n_comp = 2
  features = np.array([5, 10, 25, 50])
  results = []
  for c in features:
    #print(f'processing {c}')
    X = get_data(datasets, samples, c)
    t_batched = my_batched_pca(X, N_repeat, n_comp)
    t_reg = my_pca(X, N_repeat, n_comp)
    t_sk = np_pca(X, N_repeat, n_comp)
    results.append([t_batched, t_reg, t_sk])

  results = np.array(results)
  plt.plot(features, results[:, 0], label='batched')
  plt.plot(features, results[:, 1], label='trivial')
  plt.plot(features, results[:, 2], label='sklearn')
  plt.xlabel('number of features')
  plt.ylabel('processing time')
  plt.title(f'# of datasets = {datasets}\n# of samples {samples}')
  plt.legend()
  plt.show()

feature_count_analysis()