# get number of convolution layer in PyTorch pretrained models
# Author: Sehyun Kim
# Email: sehyun.seankim@gmail.com

import torch
import torchvision
from modified_network import ModifiedNetwork

def get_num_conv(model, cnt_conv=0):
    for layer in model.children():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            cnt_conv += 1
        elif isinstance(layer, torch.nn.modules.container.Sequential) or \
            isinstance(layer, torchvision.models.resnet.Bottleneck):
            cnt_conv = get_num_conv(layer, cnt_conv)
    return cnt_conv


def get_num_layer(model, cnt=0):
    for layer in model.children():
        if isinstance(layer, torch.nn.modules.container.Sequential) or \
            isinstance(layer, torchvision.models.resnet.Bottleneck):
            cnt = get_num_conv(layer, cnt)
        else:
            cnt += 1
    return cnt


# test_net = ModifiedNetwork(db_ind=2, db_attr=None)
# print(get_num_conv(test_net))
test_net = ModifiedNetwork(db_ind=2, db_attr=None)
print(get_num_layer(test_net))