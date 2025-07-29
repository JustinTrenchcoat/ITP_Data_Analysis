'''
This section would mainly focus on making cluster analysis for staircase/non-staircase, or specific features.
It aims to extract key factors that determines the occurance of staircase/other features
'''
import pandas as pd
from helper import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
import pickle
from scipy.cluster.hierarchy import (
    average,
    complete,
    dendrogram,
    fcluster,
    single,
    ward,
)

with open('grouped.pkl', 'rb') as f:
    groupedYears = pickle.load(f)

firstDF = groupedYears[0]
testDF = firstDF[firstDF["systemNum"] == 13]
testDF = testDF[testDF["profileNum"] == 1].copy()
print(testDF.head())
testDF = testDF[["depth", "dT/dZ"]]
print(testDF.head())

X = StandardScaler().fit_transform(testDF)
# X = testDF.copy()
# linkage_array = ward(X)


from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.datasets import make_blobs


def plot(X, labels, probabilities=None, parameters=None, ground_truth=False, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # The probability of a point belonging to its labeled cluster determines
    # the size of its marker
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_index = (labels == k).nonzero()[0]

        for ci in class_index:
            ax.plot(
                X[ci, 0],
                X[ci, 1],
                "x" if k == -1 else "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=4 if k == -1 else 1 + 5 * proba_map[ci],
            )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    preamble = "True" if ground_truth else "Estimated"
    title = f"{preamble} number of clusters: {n_clusters_}"
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    ax.set_title(title)
    plt.tight_layout()



# centers = [[1, 1], [-1, -1], [1.5, -1.5]]
# X, labels_true = make_blobs(
#     n_samples=750, centers=centers, cluster_std=[0.4, 0.1, 0.75], random_state=0
# )
# plot(X, labels=labels_true, ground_truth=True)
# plt.show()
# plt.close()

fig, axes = plt.subplots(3, 1, figsize=(10, 12))
dbs = DBSCAN(eps=0.05)
for idx, scale in enumerate([0.01, 1, 3]):
    dbs.fit(X * scale)
    if scale == 1:
        plot(X * scale, dbs.labels_, parameters={"scale": scale, "eps": 0.01}, ax=axes[idx])
plt.show()
plt.close()