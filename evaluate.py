# !/usr/local/bin/python3
# -*- coding: utf-8 -*-
from utils.header import RESULT_ROOT, DATASET_DICT, MODEL_DICT

import os
import scipy.io
import argparse
import torch
import numpy as np
from utils.distance import compute_dist

######################################################################
# Settings
# --------
USE_GPU = True
DATASET_CHOICE = 0      # 0:MARKET,    1:DUKE,         2:CUHK03-detected, 3:CUHK03-labeled
MODEL_CHOICE = 3        # 0:ResNet50,  1:DenseNet121,  2:DPN68b           3:DPN92

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--data_choice', default=DATASET_CHOICE, type=int, help='DATASET_CHOICE')
parser.add_argument('--model_choice', default=MODEL_CHOICE, type=int, help='MODEL_CHOICE')
parser.add_argument('--method', default='euclidean', type=str, help='DISTANCE MEASURE METHOD')
opt = parser.parse_args()

MODEL_NAME = MODEL_DICT[opt.model_choice]
DATASET_NAME = DATASET_DICT[opt.data_choice]
result_dir = os.path.join(RESULT_ROOT, DATASET_NAME, MODEL_NAME)
#######################################################################
# Evaluate
def evaluate(qf,ql,qc,gf,gl,gc,type='euclidean'):
    qf = qf[np.newaxis,:]
    # qf.shape = (1,2048)
    # gf.shape = (19732,2048)
    score = compute_dist(qf, gf, type=type) # cosine
    # predict index
    score = np.squeeze(score,axis=0)    # score.shape = (19732,)
    index = np.argsort(score)   # from small to large
    if type == 'cosine':
        index = index[::-1]         # from large to small
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)                           # junk bboxs index
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1  # shifted step function
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)  # TP/(TP+FP)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc


######################################################################
result = scipy.io.loadmat(os.path.join(result_dir, 'query_feature.mat'))
query_feature = result['query_f']
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
result = scipy.io.loadmat(os.path.join(result_dir, 'gallery_feature.mat'))
gallery_feature = result['gallery_f']
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
#print(query_label)
for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i],
                                    gallery_feature, gallery_label, gallery_cam,
                               type=opt.method,
                               )
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    print("%d / %d" % (i, len(query_label)))
    # print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
print('top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
