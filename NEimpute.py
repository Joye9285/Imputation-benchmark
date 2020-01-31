import numpy as np
import pandas as pd
import scipy.io as scio
import h5py
import sys
import datetime

open_file = sys.argv[1]
data_file = sys.argv[2]
save_file = sys.argv[3]

dataset_len = len(data_file.split('/')[-1])
dir_path = data_file[:-dataset_len]
cell_file = dir_path + 'cell_filtered.txt'
gene_file = dir_path + 'gene_filtered.txt'

cells = np.loadtxt(cell_file, dtype=str, delimiter='\n')
genes = np.loadtxt(gene_file, dtype=str, delimiter='\n')

V7 = True

path = open_file

if V7 == True:
    matrix = {}
    with h5py.File(path,'r') as f:
        for k, v in f.items():
            matrix[k] = np.array(v)
else:
    matrix = scio.loadmat(path)

print('start:',datetime.datetime.now())
matrix = matrix['W_singlecell_NE']

sim_sh = 1.5
f_sh = 0.01
print('matrix load finished')



for i in range(len(matrix)):
	matrix[i][i] = sim_sh * max(matrix[i])

Mnorm = matrix.copy()
for i in range(len(Mnorm)):
	Mnorm[i] = Mnorm[i]/(sum(Mnorm[i]))

path = data_file
file_type = path.split('.')[-1]
if file_type == 'tsv':
    data = pd.read_csv(path, index_col=0, sep='\t').values.T
else:
    data = pd.read_csv(path).values
print('data load finished')
print(data.shape)

data_impute = Mnorm.dot(data)
'''
for i in range(len(data_impute)):
    r = np.max(data[i])/np.max(data_impute[i])
    data_impute[i] = data_impute[i] * r

mask = data.copy()
mask[mask > 0] = -1
mask = mask + 1
'''
impute = data_impute
#impute = mask * data_impute + data

print('imputed')
print('end:', datetime.datetime.now())

path = save_file

impute = impute.T
impute = pd.DataFrame(impute, index=genes, columns=cells)
impute.to_csv(path, index=True, header=True, sep='\t')
print('save imputed data finished')
print(datetime.datetime.now(),'NE end')
