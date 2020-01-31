import pandas as pd
import numpy as np
import datetime
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.patches as mpatches
import os
import datetime

#algos = ['raw']
#algos = ['scScope']
algos = ['raw', 'SAVER', 'NE', 'scImpute', 'DrImpute', 'ZINBWaVE', 'DCA', 'scVI', 'MAGIC', 'scScope', 'SIMLR']

COLOR_ON = 'darkred'
COLOR_MID = 'lightcoral'
COLOR_OFF = 'lightgray'


#SH_ON = np.log2(6)
#SH_OFF = np.log2(2)

patches = [mpatches.Patch(color=COLOR_ON, label='ON'),mpatches.Patch(color=COLOR_MID, label='MID'),mpatches.Patch(color=COLOR_OFF, label='OFF')]

dataset = sys.argv[1]
mode = sys.argv[2]

if dataset == 'GSE72056':
    SH_ON = np.log2(6)
    SH_OFF = np.log2(2)
elif dataset == 'GSE70580':
    SH_ON = 110.0
    SH_OFF = 90.0
elif dataset == 'PBMC':
    SH_ON = 1.0
    SH_OFF = 0.4
else:
    SH_ON = 6
    SH_OFF = 2
print('shreshold', SH_OFF, SH_ON)

# get the marker file of this dataset
r_path = "marker/"+dataset+"/marker.txt"
with open(r_path) as f:
    data = list(f)
for i in range(len(data)):
    data[i] = data[i].split('\n')[0].split(',')

stat_mat = np.zeros([len(algos), 2])
stat_columns = ['plan1_2D','plan1_hD']
stat_index = algos

print('-----------start------------',datetime.datetime.now())

for i_algo in range(len(algos)):
    algo = algos[i_algo]
    print(algo, 'start')

    # get the point positions in 2D space
    pos_path = "pos/"+dataset+"/"+algo+"_"+mode+"_log.txt"
    if os.path.exists(pos_path) == False:
        print(' ---- ERROR ----:pos_path not exists', pos_path)
        continue
    data_pos = np.loadtxt(pos_path)
    
    # get the expression matrix
    if algo == 'raw':
        if mode == 'raw':
            mat_path = "../data/"+dataset+"/data_filtered.tsv"
        else:
            mat_path = "../data/"+dataset+"/data_norm_filtered.tsv"
    else:
        mat_path = "../../../../benchmarking/"+algo+"/"+mode+"/"+dataset+"_imputed.tsv"
    if os.path.exists(mat_path) == False:
        print(' ----ERROR----:mat_path not exists', mat_path)
        continue
    data_mat = pd.read_csv(mat_path, index_col=0, sep='\t')
    print(' data loaded')
    gene = data_mat.index.values
    gene = gene.reshape(-1)
    for i in range(len(gene)):
        gene[i] = gene[i].split('-')[0]
        gene[i] = gene[i].split('_')[0]
    data_mat = data_mat.values
    point_size = 20000 // len(data_pos)
    print(' size:',point_size)
    if point_size == 0:
        point_size = 3

    # construct the membership matrix
    # initial the mem_mat, 1: i-th cell belongs to j-th cluster
    mem_mat = np.zeros([len(data_pos), len(data)]).astype(int)

    # for each cluster, highlight the cells that highly expressed marker
    for i in range(len(data)):
        print(' cluster:', data[i][0])
        # i: the i-th cluster that we focus on
        # get the markers of the i-th cluster
        markers = np.array(data[i])[1:]
        index = []
        for k in range(len(markers)):
            print('     marker:', markers[k])
            flag = 0
            for j in range(len(gene)):
                if markers[k] == gene[j]:
                    index.append(j)
                    print('         line:',j,',gene:',gene[j])
                    flag = 1
                    #break
            if flag == 0:
                print('         marker not found')
        index = np.array(index)
        print('         index',index)
        index = index.astype(int)
        data_fil = data_mat[index]
        mean_ex = np.mean(data_fil, axis=0)
        # get the color of each point
        stat = [0,0,0]
        Pcolor = []
        celltype = []
        for j in range(len(mean_ex)):
            if mean_ex[j] >= SH_ON:
                Pcolor.append(COLOR_ON)
                stat[0] = stat[0] + 1
                mem_mat[j][i] = 1
                celltype.append(1)
            elif mean_ex[j] <= SH_OFF:
                Pcolor.append(COLOR_OFF)
                stat[2] = stat[2] + 1
                mem_mat[j][i] = 0
                celltype.append(2)
            else:
                Pcolor.append(COLOR_MID)
                stat[1] = stat[1] + 1
                mem_mat[j][i] = 1
                celltype.append(1)
        # calculate the index
        celltype = np.array(celltype)

        '''
        # calculate the silhouette of multi-class data
        if len(set(celltype)) > 1:
            sil_2D = metrics.silhouette_score(data_pos, celltype, metric='euclidean')
            sil_hD = metrics.silhouette_score(data_mat.T, celltype, metric='euclidean')
        else:
            sil_2D = 0
            sil_hD = 0
        stat1[i_algo][i] = sil_2D
        stat2[i_algo][i] = sil_hD
        print('         sil2D:',sil_2D,', silhD:',sil_hD)
        '''

        '''
        # visualize
        fig=plt.figure(figsize=(10,10))
        plt.scatter(data_pos[:,0],data_pos[:,1],color=Pcolor,s=point_size)
        ax=plt.gca()
        ax.legend(handles=patches,loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0., fontsize=20)
        # save the picture
        t_path = "marker/"+dataset+"/"+mode+"_"+algo+"_"+data[i][0]+".png"
        fig.savefig(t_path,bbox_inches='tight',pad_inches=0.5)
        '''
        print('         stat:', np.array(stat).astype(str))
        print('     finished')

    # output the proportion of unlabeled cells
    #print(' -----plan1 start------------',datetime.datetime.now())
    label_sum = np.sum(mem_mat, axis=1)
    label_stat = []
    for i in range(len(mem_mat[0])):
        label_stat.append(len(np.argwhere(label_sum == i).reshape(-1))/len(label_sum))
    print(' label proportion(0/1/2/3):', label_stat)

    # Plan1-PointView: create new dataset, calculate the silhouette
    # create new data_mat and new data_pos
    print(' -----plan1 start------------',datetime.datetime.now())
    new_data_mat = np.array([-1])
    new_data_pos = np.array([-1])
    new_label = []
    for i in range(len(mem_mat)):
        for j in range(len(mem_mat[0])):
            if mem_mat[i][j] == 1:
                if new_data_pos.shape == (1,):
                    #new_data_mat = data_mat.T[i].reshape(1,-1)
                    new_data_pos = data_pos[i].reshape(1,-1)
                else:
                    #new_data_mat = np.concatenate((new_data_mat, data_mat.T[i].reshape(1,-1)), axis=0)
                    new_data_pos = np.concatenate((new_data_pos, data_pos[i].reshape(1,-1)), axis=0)
                new_label.append(j)
    new_label = np.array(new_label)
    # calculate the silhouette of two dataset
    if len(set(new_label)) <= 1:
        stat_mat[i_algo] = np.array([-1,-1])
        continue
    plan1_2D = metrics.silhouette_score(new_data_pos, new_label, metric='euclidean')
    plan1_hD = -1
    #plan1_hD = metrics.silhouette_score(new_data_mat, new_label, metric='euclidean')

    # Plan2-ClassView
    # for each cluster, calculate the intra-dis and the inter-dis
    print(' -----plan2 start------------',datetime.datetime.now())
    '''
    # calculate the whole distance matrix
    DIS_MAT = np.zeros([len(data_pos), len(data_pos)])
    DIS_POS = np.zeros([len(data_pos), len(data_pos)])
    for i_x in range(len(data_pos)):
        for j_x in range(len(data_pos)):
            DIS_MAT[i_x][j_x] = np.linalg.norm()
    '''
    #plan2_2D = 0
    #plan2_hD = 0
    '''
    intra_dis_2D = []
    intra_dis_hD = []
    clus_med_2D = np.zeros([len(mem_mat[0]), len(data_mat)])
    clus_med_hD = np.zeros([len(mem_mat[0]), 2])
    inter_dis_2D = []
    inter_dis_hD = []
    for i in range(len(mem_mat[0])):
        # for the i-th cluster
        # the intra-dis
        cell_index = np.argwhere(mem_mat[:,i] == 1).reshape(-1)
        new_data_mat = data_mat.T[cell_index]
        new_data_pos = data_pos[cell_index]
        dis_mat = np.zeros([len(cell_index), len(cell_index)])
        dis_pos = np.zeros([len(cell_index), len(cell_index)])
        for i_x in range(len(cell_index)):
            for j_x in range(len(cell_index)):
                dis_mat[i_x][j_x] = np.linalg.norm(new_data_mat[i_x] - new_data_mat[j_x])
                dis_pos[i_x][j_x] = np.linalg.norm(new_data_pos[i_x] - new_data_pos[j_x])
        intra_dis_2D.append(np.mean(dis_pos))
        intra_dis_hD.append(np.mean(dis_mat))
        clus_med_2D[i] = np.mean(dis_pos)
        clus_med_hD[i] = np.mean(dis_mat)
    for i in range(len(mem_mat[0])):
        min_dist_2D = 0
        min_dist_hD = 0
        for j in range(len(mem_mat[0])):
            if i == j:
                continue
            dist_2D = np.linalg.norm(clus_med_2D[i] - clus_med_2D[j])
            dist_hD = np.linalg.norm(clus_med_hD[i] - clus_med_hD[j])
            if min_dist_2D == 0:
                min_dist_2D = dist_2D
                min_dist_hD = dist_hD
            else:
                if min_dist_2D > dist_2D:
                    min_dist_2D = dist_2D
                if min_dist_hD > dist_hD:
                    min_dist_hD = dist_hD
        inter_dis_2D.append(min_dist_2D)
        inter_dis_hD.append(min_dist_hD)
    inter_dis_2D = np.array(inter_dis_2D)
    inter_dis_hD = np.array(inter_dis_hD)
    intra_dis_2D = np.array(intra_dis_2D)
    intra_dis_hD = np.array(intra_dis_hD)
    # calculate the mean dist
    #plan2_2D = 0
    #plan2_hD = 0
    for i in range(len(inter_dis_2D)):
        plan2_2D = plan2_2D + (inter_dis_2D[i] - intra_dis_2D[i])/max(inter_dis_2D[i], intra_dis_2D[i])
        plan2_hD = plan2_hD + (inter_dis_hD[i] - intra_dis_hD[i])/max(inter_dis_hD[i], intra_dis_hD[i])
    plan2_2D = plan2_2D / len(inter_dis_2D)
    plan2_hD = plan2_hD / len(inter_dis_hD)
    '''
    print(' -----plan2 end--------------',datetime.datetime.now())
    
    stat_mat[i_algo] = np.array([plan1_2D,plan1_hD])
    print('-- stat --:', stat_mat[i_algo])

stat_mat = pd.DataFrame(stat_mat, columns=stat_columns, index=stat_index)
stat_path = "marker/"+dataset+"/stat_"+mode+"_plan1.csv"
stat_mat.to_csv(stat_path)

'''
stat1 = pd.DataFrame(stat1, index=stat_index, columns=stat_column)
stat1.to_csv(stat1_path)
stat2 = pd.DataFrame(stat2, index=stat_index, columns=stat_column)
stat2.to_csv(stat2_path)
'''
