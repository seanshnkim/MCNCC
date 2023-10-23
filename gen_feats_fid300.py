import numpy as np
import cv2
import os
from scipy.io import savemat
from utils_custom.get_db_attrs import get_db_attrs
from generate_db_CNNfeats import generate_db_CNNfeats
import pickle

from modified_network import ModifiedNetwork



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

    # net = load_modify_network(db_ind, db_attr)  
    net = ModifiedNetwork(db_ind=2, db_attr=db_attr)
    
    #REVIEW In this case, net.vars(1).name is same as 'data'
    ims_transposed = ims.transpose(0, 2, 1, 3)
    all_db_feats = generate_db_CNNfeats(net, ims_transposed)  
    all_db_labels = treadids.reshape(1, 1, 1, -1)

    
    # feat_idx = 27
    feat_dims = all_db_feats.shape
    rfsIm = ...
    
    # Creating a directory
    output_dir = os.path.join('feats', dbname)
    os.makedirs(output_dir, exist_ok=True)

    # Saving variables
    for i in range(all_db_feats.shape[3]):
        db_feats = all_db_feats[:, :, :, i]
        db_labels = all_db_labels[:, :, :, i]
        
        # Saving the first index with additional variables
        if i == 0:
            save_path = os.path.join(output_dir, 'fid300_001.pkl')
            with open(save_path, 'wb') as file:
                pickle.dump({
                    'db_feats': db_feats,
                    'db_labels': db_labels,
                    'feat_dims': feat_dims,
                    # 'rfsIm': rfsIm,
                    # 'trace_H': trace_H,
                    # 'trace_W': trace_W
                }, file)
        else:
            save_path = os.path.join(output_dir, f'fid300_{i+1:03d}.pkl')
            with open(save_path, 'wb') as file:
                pickle.dump({
                    'db_feats': db_feats,
                    'db_labels': db_labels
                }, file)
