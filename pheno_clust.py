import numpy as np
import pandas as pd
import phenograph
import sys
import os
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics

#path = sys.argv[1]
#algo = path.split('/')[0]
#dataset = path.split('/')[1]
'''
algo = sys.argv[1]
dataset = sys.argv[2]
mode = sys.argv[3]

if mode == '1':
    read_path = algo + '/raw/' + dataset + "_imputed.tsv"
    save_path = algo + "/pheno/raw/" + dataset + '.txt'
elif mode == '2':
    read_path = algo + '/' + dataset + "_imputed.tsv"
    save_path = algo + "/pheno/" + dataset + '.txt'
else:
    read_path = "data/" + dataset + "/data_filtered_1000.tsv"
    save_path = "Raw/raw/pheno/" + dataset + '.txt'
#save_path = algo + "/pheno/raw/" + dataset + '.txt'
print('\n')
print(algo, dataset)
'''

read_path = sys.argv[1]
save_path = sys.argv[2]
print('\n')
print(read_path)

data = pd.read_csv(read_path, index_col=0, sep='\t')
cell = np.array(data.columns)
print(len(cell))
data = data.values.T

data = np.log2(data + 1)
data = PCA(n_components=50).fit_transform(data)

label, _, _ = phenograph.cluster(data)
print(len(label))
result = []
for i in range(len(cell)):
    result.append([cell[i], label[i]])
result = np.array(result)

df = pd.DataFrame(result, columns=['cell','label'])
df.to_csv(save_path, index=False, sep='\t')
