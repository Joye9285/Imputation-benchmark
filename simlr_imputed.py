import numpy as np
from scipy import sparse, spatial
import pandas as pd
import tasklogger
import os
import sys
import datetime

if len(sys.argv) != 3:
    print('usage: python xxx.py dataset corrupt')
    exit(0)

prefix = '/gpfs/share/home/1210305123/chengyi/BenchMarking/Bioinformatics/data/'
dataset = sys.argv[1]
corrupt = int(sys.argv[2])

print('_______________________________________________')
print('_______________________________________________')
print(dataset,'SIMLR')

def _diff_op(dataset,corrupt):
    if corrupt == 0:
        data_path = 'SIMLR/' + dataset + '_S.tsv'
    else:
        data_path = 'SIMLR/corruption/' + dataset + '_S.tsv'

    data1 = pd.read_csv(data_path, sep='\t')

    simlarity = data1.values[:, 1:].T.astype(np.float32)
    print('Similarity Matrix:',simlarity.shape)

    norm = np.sum(simlarity,axis = 1).reshape(-1,1)

    result = simlarity / norm

    return result



def _calculate_error(data, data_prev=None, weights=None,
                     subsample_genes=None):

    if subsample_genes is not None:
        data = data[:, subsample_genes]
    if weights is None:
        weights = np.ones(data.shape[1]) / data.shape[1]
    if data_prev is not None:
        _, _, error = spatial.procrustes(data_prev, data)
    else:
        error = 99999
    return error, data


def _imputed(dataset,corrupt):
    if corrupt == 0:
        data_path = prefix + dataset + '/data_filtered.tsv'
        save_path = 'SIMLR/imputed/'+dataset+'_SIMLR_imputed.tsv'
    else:
        data_path = prefix + dataset + '/data_filtered_c.tsv'
        save_path = 'SIMLR/imputed/corruption/'+dataset+'_SIMLR_imputed.tsv'

    data1 = pd.read_csv(data_path, sep='\t')

    cell = data1.columns.values

    data_gene = data1.values

    gene = data_gene[:, 0]
    gene = gene.reshape([-1, 1])

    data = data_gene[:, 1:]

    data_imputed = data.T.astype(np.float32)

    print('raw data:',data_imputed.shape)

    t_opt = None
    t_max = 20
    threshold = 0.001

    diff_op = _diff_op(dataset,corrupt)

    # i = 0
    # data_prev = None
    # while (t_opt is None and i < t_max) or \
    #         (t_opt is not None and i < t_opt):
    #     i += 1
    #     data_imputed = diff_op.dot(data_imputed)
    #     # print(data_imputed.shape)
    #     error, data_prev = _calculate_error(
    #         data_imputed, data_prev)
    #     if error < threshold and t_opt is None:
    #         t_opt = i + 1
    #         tasklogger.log_info(
    #             "Automatically selected t = {}".format(t_opt))

    data_imputed = diff_op.dot(data_imputed)

    data_imputed = data_imputed.T
    data_imputed = np.concatenate([gene, data_imputed], axis=1)
    print('data_imputed', data_imputed.shape)


    save = pd.DataFrame(data_imputed, columns=cell)
    save.to_csv(save_path, sep='\t', index=False)


_imputed(dataset,corrupt)
