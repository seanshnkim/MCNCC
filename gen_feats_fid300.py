import numpy as np
import cv2
import os
from scipy.io import savemat
from utils_custom.get_db_attrs import get_db_attrs

import torch
import torch.nn as nn
from torchvision import models


# Load and modify network
class ModifiedNetwork(nn.Module):
    def __init__(self, db_ind, db_attr):
        super(ModifiedNetwork, self).__init__()
        
        if db_ind == 0:
            # Add identity layer (equivalent operation in PyTorch)
            self.layer = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
            self.layer.weight.data = torch.tensor([[[[1]], [[0]], [[0]]]], dtype=torch.float32)
        else:
            # Load pre-trained model and modify it
            # model_path = os.path.join('models', db_attr[2])
            # pre_trained_model = torch.load(model_path)
            #REVIEW - 현재로서는 모델 파일 자체를 불러올 게 아니라 그냥 torchvision에서 다운로드받자
            pretrained_model = models.resnet50(pretrained=True)
            
            # Remove layers after the specified index
            layer_index = db_attr[0]  # You might need to adjust how you get this index based on the actual attribute
            modified_pretrained_layers = list(pretrained_model.children())[:layer_index]
            self.model = nn.Sequential(*modified_pretrained_layers)
            
    def forward(self, x):
        if hasattr(self, 'layer'):
            return self.layer(x)
        else:
            return self.model(x)

# You can now instantiate and use ModifiedNetwork by passing the appropriate db_ind and db_attr



def gen_feats_fid300(db_ind=2):
    imscale = 0.5

    db_attr, _, dbname = get_db_attrs('fid300', db_ind)  # You should define get_db_attrs function

    ims = []
    for i in range(1, 1176):
        im = cv2.imread(os.path.join('datasets', 'FID-300', 'references', f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        # only 4 shoes are 1 pixel taller (height=587)
        im = im[0:586, :]

        if i in [107, 525]:
            im = cv2.flip(im, 1)

        w = im.shape[1]
        pad_left = np.full((im.shape[0], (270 - w) // 2), 255, dtype=np.uint8)
        pad_right = np.full((im.shape[0], (270 - w + 1) // 2), 255, dtype=np.uint8)
        
        im = np.hstack((pad_left, im, pad_right))
        im = cv2.resize(im, None, fx=imscale, fy=imscale, interpolation=cv2.INTER_AREA)
        
        ims.append(im)

    ims = np.array(ims)
    mean_im = np.mean(ims, axis=0)
    mean_im_pix = np.mean(mean_im)
    ims = ims - mean_im_pix
    ims = np.tile(ims[:, :, np.newaxis], (1, 1, 3, 1))

    groups = [
        [162, 390, 881],
        [661, 662, 1023],
        [24, 701],
        [25, 604],
        [35, 89],
        [45, 957],
        [87, 433],
        [115, 1075],
        [160, 1074],
        [196, 813],
        [270, 1053],
        [278, 1064],
        [306, 828],
        [363, 930],
        [453, 455],
        [656, 788],
        [672, 687],
        [867, 1015],
        [902, 1052],
        [906, 1041],
        [1018, 1146],
        [1065, 1162],
        [1156, 1157],
        [1169, 1170],
    ]

    treadids = np.zeros(1175)
    id = 0
    for g in groups:
        id += 1
        for p in g:
            treadids[p-1] = id
    
    for p in range(1175):
        if treadids[p] == 0:
            id += 1
            treadids[p] = id

    # net = load_modify_network(db_ind, db_attr)  # You should define load_modify_network function
    net = ModifiedNetwork(db_ind=2, db_attr=db_attr)
    
    
    all_db_feats, all_db_labels = generate_db(net, ims, treadids)  # You should define generate_db function

    save_features(dbname, all_db_feats, all_db_labels)  # You should define save_features function


# You may need to define or modify the following functions based on your specific needs and libraries:
# - get_db_attrs
# - load_modify_network
# - generate_db
# - save_features
