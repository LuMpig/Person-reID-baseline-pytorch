# !/usr/local/bin/python3
# -*- coding: utf-8 -*-
from utils.header import DATA_ROOT, MODEL_ROOT, RESULT_ROOT, \
    DATASET_DICT, MODEL_DICT, NUM_CLASS_TRAIN

import os
import argparse
import scipy.io

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from net import get_net
######################################################################
# Settings
# --------
USE_GPU = True
DATASET_CHOICE = 0      # 0:MARKET,    1:DUKE,         2:CUHK03-detected, 3:CUHK03-labeled
MODEL_CHOICE = 3        # 0:ResNet50,  1:DenseNet121,  2:DPN68b           3:DPN92
BATCH_SIZE = 8
NUM_WORKERS = 1

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--data_choice', default=DATASET_CHOICE, type=int, help='DATASET_CHOICE')
parser.add_argument('--model_choice', default=MODEL_CHOICE, type=int, help='MODEL_CHOICE')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='batchsize')
parser.add_argument('--num_workers', default=1, type=int, help='num_workers')
opt = parser.parse_args()

MODEL_NAME = MODEL_DICT[opt.model_choice]
DATASET_NAME = DATASET_DICT[opt.data_choice]

data_dir = os.path.join(DATA_ROOT, DATASET_NAME)
model_dir = os.path.join(MODEL_ROOT, DATASET_NAME, MODEL_NAME)
result_dir = os.path.join(RESULT_ROOT, DATASET_NAME, MODEL_NAME)
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

create_net = get_net(opt.model_choice)

######################################################################
# Load Data
# ---------
data_transforms = transforms.Compose([
        transforms.Resize((288,144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batch_size,
                shuffle=False, num_workers=opt.num_workers) for x in ['gallery','query']}

class_names = image_datasets['query'].classes

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join(model_dir,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network

######################################################################
# Extract feature
# ----------------------
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        if MODEL_NAME == 'ResNet50':
            ff = torch.FloatTensor(n,2048).zero_()
        elif MODEL_NAME == 'DenseNet121':
            ff = torch.FloatTensor(n,1024).zero_()
        elif MODEL_NAME == 'DPN68b':
            ff = torch.FloatTensor(n,832).zero_()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            if USE_GPU:
                input_img = Variable(img.cuda())
            else:
                input_img = Variable(img)
            outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff+f
        ff = ff.div(2)
        features = torch.cat((features,ff), 0)
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


def get_id_CUHK(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        camera = 2*(int(filename.split('_')[0])-1) + int(filename.split('_')[2])
        label = path.split('/')[-2]
        labels.append(int(label))
        camera_id.append(int(camera))
    return camera_id, labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

if opt.data_choice == 2 or opt.data_choice == 3:
    gallery_cam, gallery_label = get_id_CUHK(gallery_path)
    query_cam, query_label = get_id_CUHK(query_path)
else:
    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
model_structure = create_net(NUM_CLASS_TRAIN[opt.data_choice])
model = load_network(model_structure)

# Remove the final fc layer and classifier layer
model.model.fc = nn.Sequential()
model.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if USE_GPU:
    model = model.cuda()

# gallery_feature
gallery_feature = extract_feature(model,dataloaders['gallery'])
result = {
    'gallery_f'     :   gallery_feature.numpy(),
    'gallery_label' :   gallery_label,
    'gallery_cam'   :   gallery_cam,
}
scipy.io.savemat(os.path.join(result_dir,'gallery_feature.mat'), result)

# query_feature
query_feature = extract_feature(model,dataloaders['query'])
result = {
    'query_f'       :   query_feature.numpy(),
    'query_label'   :   query_label,
    'query_cam'     :   query_cam
}
scipy.io.savemat(os.path.join(result_dir,'query_feature.mat'), result)



''' pytorch_result.mat:
gallery_cam:    1x19732      int64
gallery_f:      19732x2048   single
gallery_label:  1x19732      int64
query_cam:      1x3368       int64
query_f:        3368x2048    single
query_label:    1x3368       int64
'''
