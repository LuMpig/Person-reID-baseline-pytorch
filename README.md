# Person-reID-baseline-pytorch
This repository base on [layumi/Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch).

**What's new:** Add CUHK03 dataset.We use new test protocol from [Re-ranking person re-identification with k-reciprocal encoding](https://arxiv.org/pdf/1701.08398.pdf). You can find matlab code that transform CUHK03 dataset to pytorch format  [here](https://github.com/hyk1996/Person-reID-baseline-pytorch/tree/master/data_prepare/prepare_CUHK-03).

**What's new:** Add DPN68b and DPN92 (DPN, dual path net) as backbone network.

## Experiment result
batchsize = 16, without dropout and data augmentation, image size is 288x144 (for training and testing).

### Market-1501
| Model   | top-1   | mAP    |
| :-----: | :-----: | :----: |
| **ResNet50**    | 87.5 | 69.9 |
| **DenseNet121** | **89.9** | **73.9** |
| **DPN68b**      | 83.8 | 64.9 |

### DukeMTMC-reID
| Model   | top-1   | mAP    |
| :-----: | :-----: | :----: |
| **ResNet50**    | 76.8 | 57.0 |
| **DenseNet121** | **81.0** | **63.4** |
| **DPN68b**      | 73.2 | 52.8 |

### CUHK03-labeled
| Model   | top-1   | mAP    |
| :-----: | :-----: | :----: |
| **ResNet50**    | 44.6 | 39.9 |
| **DenseNet121** | **46.8** | **42.1** |
| **DPN68b**      | 42.8 | 38.0 |

### CUHK03-detected
| Model   | top-1   | mAP    |
| :-----: | :-----: | :----: |
| **ResNet50**    | 42.6 | 37.2 |
| **DenseNet121** | **43.1** | **37.9** |
| **DPN68b**      | 40.6 | 35.4 |
