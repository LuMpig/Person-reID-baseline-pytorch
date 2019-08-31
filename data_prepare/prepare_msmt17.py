import os
import os.path as osp
from shutil import copyfile


# You only need to change this line to your dataset download path
root = '~/path/to/MSMT17_V1'
if not os.path.isdir(root):
    print('please change the download_path')

def read_image_list(root, train_or_test, type):
    # return img path list and label list
    list_dir = osp.join(root, 'list_{}.txt'.format(type))
    data_dir = osp.join(root, train_or_test)
    with open(list_dir, 'r') as f:
        image_path = []
        labels = []
        for line in f:
            img_and_id = line.strip("\n").split(' ')
            image_path.append(osp.join(data_dir, img_and_id[0]))
            labels.append(int(img_and_id[1]))
    return image_path, labels


#------- train -------
paths, labels = read_image_list(root, 'train', 'train')
save_path = osp.join(root, 'pytorch', 'train')

for src_path, label in zip(paths, labels):
    filename = src_path.split('/')[-1]
    dst_path = osp.join(save_path, "%04d" % label)
    if not osp.isdir(dst_path):
        os.makedirs(dst_path)
    copyfile(src_path, osp.join(dst_path, filename))
    print(filename)


#------- val -------
paths, labels = read_image_list(root, 'train', 'val')
save_path = osp.join(root, 'pytorch', 'val')

for src_path, label in zip(paths, labels):
    filename = src_path.split('/')[-1]
    dst_path = osp.join(save_path, "%04d" % label)
    if not osp.isdir(dst_path):
        os.makedirs(dst_path)
    copyfile(src_path, osp.join(dst_path, filename))
    print(filename)


#------- gallery -------
paths, labels = read_image_list(root, 'test', 'gallery')
save_path = osp.join(root, 'pytorch', 'gallery')

for src_path, label in zip(paths, labels):
    filename = src_path.split('/')[-1]
    dst_path = osp.join(save_path, "%04d" % label)
    if not osp.isdir(dst_path):
        os.makedirs(dst_path)
    copyfile(src_path, osp.join(dst_path, filename))
    print(filename)


#------- query -------
paths, labels = read_image_list(root, 'test', 'query')
save_path = osp.join(root, 'pytorch', 'query')

for src_path, label in zip(paths, labels):
    filename = src_path.split('/')[-1]
    dst_path = osp.join(save_path, "%04d" % label)
    if not osp.isdir(dst_path):
        os.makedirs(dst_path)
    copyfile(src_path, osp.join(dst_path, filename))
    print(filename)