import numpy as np
import pandas as pd
import sys

read_path = sys.argv[1]
save_path = sys.argv[2]
print('start')
print(read_path)

raw_data = pd.read_csv(read_path, index_col=0, sep='\t')
data = raw_data.values.T.astype(float)
cells = raw_data.columns
genes = raw_data.index
print('data loaded')

print(data.shape)
print('old')
print(data[:10,:20])

for i in range(len(data)):
    data[i] = data[i]/sum(data[i])*100000

print('new')
print(data[:10,:20])

data = data.T.astype(float)
'''
df = pd.DataFrame(data, columns=cells, index=genes)
path = save_path + "_cpm.tsv"
df.to_csv(path, index=True, header=True, sep='\t')
print('cpm saved')
'''
datalog2 = np.log2(data + 1)
df = pd.DataFrame(datalog2, columns=cells, index=genes)
path = save_path + "data_norm_filtered.tsv"
df.to_csv(path, index=True, header=True, sep='\t')
print('log2 saved')
'''
datalog10 = np.log10(data + 1)
df = pd.DataFrame(datalog10, columns=cells, index=genes)
path = save_path + "_cpm_log10.tsv"
df.to_csv(path, index=True, header=True, sep='\t')
print('log10 saved')

print('data saved')
'''
