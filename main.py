import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from pcas import MyPCA, MyBatchedPCA, BatchedTorchPCA
import torch
import pyinstrument


def get_data(datasets=50, samples=1000, features=100):
    x = np.concatenate(
        [make_blobs(n_samples=samples, n_features=features)[0][np.newaxis, ...] for _ in range(datasets)])
    return x


def my_batched_pca(x, n, comp=2):
    # ---------------------------------------------------------
    for _ in range(n):
        MyBatchedPCA(n_components=comp).fit(x)


def my_torch_batched_pca(x, n, comp=2):
    # ---------------------------------------------------------
    # TODO: consider computing the time to device memory
    x = torch.tensor(x)#.to(0)
    for _ in range(n):
        BatchedTorchPCA(n_components=comp).fit(x)


def my_pca(x, n, comp=2):
    # ---------------------------------------------------------
    for _ in range(n):
        for dataset in x:
            MyPCA(n_components=comp).fit(dataset)


def np_pca(x, n, comp=2):
    for _ in range(n):
        for dataset in x:
            PCA(n_components=comp, svd_solver='covariance_eigh').fit(StandardScaler().fit_transform(dataset))


def feature_count_analysis():
    n_repeat = 3
    datasets = 150
    samples = 200
    n_comp = 2
    features = np.array([200])
    for c in features:
        x = get_data(datasets, samples, c)
        my_torch_batched_pca(x, n_repeat, n_comp)
        my_batched_pca(x, n_repeat, n_comp)
        my_pca(x, n_repeat, n_comp)
        np_pca(x, n_repeat, n_comp)


profiler = pyinstrument.Profiler()
profiler.start()
feature_count_analysis()
profiler.stop()
print(profiler.output_text(unicode=True, color=True, show_all=False))
