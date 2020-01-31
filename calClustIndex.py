import numpy as np
import pandas as pd
import sys
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics

pred_path = sys.argv[1]
true_path = sys.argv[2]
data_path = sys.argv[3]
print(pred_path)

pred = pd.read_csv(pred_path, sep='\t').values
pred = pred[:, -1].astype(int)

true = pd.read_csv(true_path, sep='\t').values
true = true[:, -1].astype(int)

data = pd.read_csv(data_path, index_col=0, sep='\t').values.T

print(len(pred), len(true), len(data))
ARI = adjusted_rand_score(pred, true)
silhouette_SC3 = metrics.silhouette_score(data, pred, metric='euclidean')
silhouette_grd = metrics.silhouette_score(data, true, metric='euclidean')
print('ARI value is:',ARI)
print('silhouette value is:',silhouette_SC3, silhouette_grd)
print('\n')
