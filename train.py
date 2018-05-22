# !/usr/local/bin/python3
# -*- coding: utf-8 -*-
from utils.header import DATA_ROOT, DATASET_DICT, MODEL_ROOT, MODEL_DICT

import os
import time
import argparse
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from utils.random_erasing import RandomErasing
from net import get_net

######################################################################
# Settings
# --------
USE_GPU = True
DATASET_CHOICE = 0      # 0:MARKET,    1:DUKE,         2:CUHK03-detected, 3:CUHK03-labeled
MODEL_CHOICE = 3        # 0:ResNet50,  1:DenseNet121,  2:DPN68b           3:DPN92
BATCH_SIZE = 16
NUM_EPOCH = 60

######################################################################
# Argument
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data_choice', default=DATASET_CHOICE, type=int, help='DATASET_CHOICE')
parser.add_argument('--model_choice', default=MODEL_CHOICE, type=int, help='MODEL_CHOICE')
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='batchsize')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--num_workers', default=1, type=int, help='num_workers')
opt = parser.parse_args()

MODEL_NAME = MODEL_DICT[opt.model_choice]
DATASET_NAME = DATASET_DICT[opt.data_choice]

data_dir = os.path.join(DATA_ROOT, DATASET_NAME)
model_dir = os.path.join(MODEL_ROOT, DATASET_NAME, MODEL_NAME)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

create_net = get_net(opt.model_choice)

######################################################################
# Function
# --------
def set_random_seed(seed = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join(model_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if USE_GPU:
        network.cuda()

set_random_seed()

######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join(model_dir, 'train.jpg'))


######################################################################
# Load Data
# ---------
#
transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        # transforms.Resize(144, interpolation=3),
        # transforms.RandomCrop((256, 128)),
        transforms.Resize((288, 144), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list + [RandomErasing(opt.erasing_p)]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

transform_val_list = [
        transforms.Resize(size=(288,144),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}

train_all = ''
if opt.train_all:
     train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batch_size,
                                             shuffle=True, num_workers=opt.num_workers)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

inputs, classes = next(iter(dataloaders['train']))

######################################################################
# Training the model
# ------------------
y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=60):
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for count, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                if USE_GPU:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                print('step : ({}/{})  |  loss : {:.4f}'.format(count*opt.batch_size,dataset_sizes[phase], running_loss))
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch%10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')


######################################################################
# Model and Optimizer
# ----------------------
model = create_net(len(class_names))
print(model)

if USE_GPU:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

ignored_params = list(map(id, model.model.fc.parameters() )) + list(map(id, model.classifier.parameters() ))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.SGD([
             {'params': base_params, 'lr': 0.01},
             {'params': model.model.fc.parameters(), 'lr': 0.1},
             {'params': model.classifier.parameters(), 'lr': 0.1}
         ], momentum=0.9, weight_decay=5e-4, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

######################################################################
# Main
# ----------------------

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                    num_epochs = NUM_EPOCH)
