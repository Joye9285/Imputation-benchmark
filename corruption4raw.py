import numpy as np
import sys
import pandas as pd
import scanpy.api as sc

if len(sys.argv) != 3:
    print('usage: python xxx.py path mode')
    exit(0)

print('data:', sys.argv[1])
dropout_rate = 0.1
path = sys.argv[1]
mode = sys.argv[2]

# read data
#if mode == '2':
#    data_raw = pd.read_csv(path + "/data_filtered_1000.tsv", sep='\t', index_col=0)
#    print('data_raw loaded')
data = pd.read_csv(path + "/data_filtered_1000.tsv", sep='\t', index_col=0)
print('data loaded')
col = data.columns
idx = data.index

data = data.values.T
#if mode == '2':
#    data_raw = data_raw.values.T
#data = pd.read_csv(sys.argv[1]).values
mask = np.zeros(data.shape)

for i in range(len(data)):
    for j in range(len(data[i])):
        # only consider nonzero-data
        if data[i][j] > 1e-2:
            r = np.random.rand()
            if r < dropout_rate:
                mask[i][j] = 1
                data[i][j] = 0
                
                #if mode == '2':
                #    data_raw[i][j] = 0
print('mask finished')

data = data.T
#if mode == '2':
#    data_raw = data_raw.T

'''
if mode == '2':
    # 存csv
    dff = pd.DataFrame(data_raw, columns=col, index=idx)
    #path = sys.argv[2].split('.')[0] + '.csv'
    dff.to_csv(path+"/data_filtered_c.csv", index=True, header=True)
    dff.to_csv(path+"/data_filtered_c.tsv", index=True, header=True, sep='\t')
    print('data shape:', dff.values.shape)
    print('data_raw saved')
'''

df = pd.DataFrame(data, columns=col, index=idx)
df.to_csv(path+"/data_filtered_c.tsv", index=True, header=True, sep='\t')
print('data shape:', df.values.shape)
print('data saved')

mask = pd.DataFrame(mask)
mask.to_csv(path + '/data.mask', index=False)
print('mask saved')
