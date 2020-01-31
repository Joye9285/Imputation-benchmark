import pandas as pd
import numpy as np
import sys

data_path = sys.argv[1]
save_path = sys.argv[1].split('.')[0] + "_filtered.tsv"
dir_path = sys.argv[2]
print(data_path)

cell_path = dir_path + "/cell_index.txt"
gene_path = dir_path + "/gene_index.txt"
type_path = dir_path + "/subtype.txt"
new_type_path = dir_path + "/subtype_filtered.txt"
#print(data_path)

data = pd.read_csv(data_path, index_col=0, sep='\t')
cells = np.array(data.columns)
genes = np.array(data.index)
data = data.values.T
print('data loaded')
print('old data shape', data.shape)

nGene = []
for i in range(len(data)):
    nGene.append(len(np.argwhere(data[i] > 0)))
nGene = np.array(nGene)
Q1 = np.percentile(nGene, 25)
Q3 = np.percentile(nGene, 75)
IQR = Q3 - Q1
high_v = Q3 + 3 * IQR
low_v = Q1 - 3 * IQR
x = np.argwhere(nGene <= high_v)
y = np.argwhere(nGene >= low_v)
index1 = np.intersect1d(x, y)
#index1 = np.argwhere(nGene <= high_v and nGene >= low_v)

nUMI = np.sum(data, axis=1)
Q1 = np.percentile(nUMI, 25)
Q3 = np.percentile(nUMI, 75)
IQR = Q3 - Q1
high_v = Q3 + 3 * IQR
low_v = Q1 - 3 * IQR
x = np.argwhere(nUMI <= high_v)
y = np.argwhere(nUMI >= low_v)
index2 = np.intersect1d(x, y)
#index2 = np.argwhere(nUMI <= high_v and nUMI >= low_v)

index = []
for i in range(len(index1)):
    if index1[i] in index2:
        index.append(index1[i])
index = np.array(index)
index = index.reshape(len(index))

data = data[index]
cells = cells[index]
data = data.T
np.savetxt(cell_path, index, fmt='%d')

subtype = np.loadtxt(type_path, dtype=str)
subtype = subtype[index]
np.savetxt(new_type_path, subtype, fmt='%s')

nCell = []
for i in range(len(data)):
    nCell.append(len(np.argwhere(data[i] > 0)))
nCell = np.array(nCell)
index = np.argwhere(nCell >= 3)
index = index.reshape(len(index))
np.savetxt(gene_path, index, fmt='%d')

data = data[index]
genes = genes[index]
print('filtering finished', data.shape)

df = pd.DataFrame(data, index=genes, columns=cells)
df.to_csv(save_path, index=True, header=True, sep='\t')
