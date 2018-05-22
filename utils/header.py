import os

DATA_ROOT = '/home/kevinh/Dataset/reid/pytorch'
MODEL_ROOT = './model'
RESULT_ROOT = './result'

DATASET_DICT = {
    0  :  'Market-1501',
    1  :  'dukeMTMC-reID',
    2  :  'CUHK-03/detected',
    3  :  'CUHK-03/labeled',
}

MODEL_DICT = {
    0  :  'ResNet50',
    1  :  'DenseNet121',
    2  :  'DPN68b',
    3  :  'DPN92',
}

NUM_CLASS_TRAIN = [751,702,767,767,]
