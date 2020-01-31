import numpy as np
import pandas as pd
import scipy.io as scio
import sys
import datetime

open_file = sys.argv[1]
save_file = sys.argv[2]

path = open_file
file_type = path.split('.')[-1]

#print(datetime.datetime.now(),'NE started')

if file_type == 'txt':
    data = np.loadtxt(path)
elif file_type == 'csv':
    data = pd.read_csv(path).values
else:
    data = pd.read_csv(path, sep='\t', index_col=0).values.T
print('file load finished')

print(datetime.datetime.now(),'NE started')

# add the normalization step here
data = data.astype(float)

for i in range(len(data)):
    data[i] = data[i]/sum(data[i])*100000

data = np.log2(data + 1)
print('normalization finished')
#print(datetime.datetime.now(),'NE started')

co_matrix = np.corrcoef(data)
for i in range(len(co_matrix)):
    co_matrix[i][i] = 0

print('co_matrix calculated')

path = "data/network.mat"
NEdata = scio.loadmat(path)
NEdata['W_singlecell'] = co_matrix

path = save_file
scio.savemat(path, NEdata, do_compression='True')
print('file save finished')
