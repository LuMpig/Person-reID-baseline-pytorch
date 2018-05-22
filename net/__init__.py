from .resnet50 import reid_resnet50 as ResNet50
from .densenet121 import reid_densenet121 as DenseNet121
from .dpn68b import reid_dpn68b as DPN68b
from .dpn92 import reid_dpn92 as DPN92

__factory = {
    0   :  ResNet50,
    1   :  DenseNet121,
    2   :  DPN68b,
    3   :  DPN92,
}

def get_net(choice):
    return __factory[choice]