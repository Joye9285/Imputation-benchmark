import numpy as np
import sys
import pandas as pd


'''
note:   tsv0 is the matrix before corruption
        tsv1 is imputed matrix
'''

if len(sys.argv) != 4:
    print('usage: python xxx.py tsv0 tsv1 mask')
    exit(0)

print(sys.argv[2])

ftype = sys.argv[2].split('.')[-1]
if ftype == 'tsv':
    data0 = pd.read_csv(sys.argv[1],sep='\t',index_col=0).values.T
    print('corrupted data:', data0.shape)
    data1 = pd.read_csv(sys.argv[2],sep='\t',index_col=0).values.T
    print('imputed data:', data1.shape)
    mask = pd.read_csv(sys.argv[3]).values
    print('mask:', mask.shape)
else:
    data0 = pd.read_csv(sys.argv[1]).values
    print('corrupted data:', data0.shape)
    data1 = pd.read_csv(sys.argv[2]).values
    print('imputed data:', data1.shape)
    mask = pd.read_csv(sys.argv[3]).values
    print('mask:', mask.shape)

assert(mask.shape == data0.shape)
assert(mask.shape == data1.shape)

row, col = mask.shape 

sum_err = 0
sum_exp = 0

data0 = data0 * mask
data1 = data1 * mask

x = np.abs(data0 - data1)

sum_err = np.sum(np.abs(data0 - data1))
sum_exp = len(np.argwhere(mask > 0))

impu_err = sum_err/sum_exp
print('data name:', sys.argv[2])
print('imputation error = ', impu_err)
print('\n')
