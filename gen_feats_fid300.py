import numpy as np
import cv2
import os
import pickle

from modified_network import ResNet50Encoder
from getVarReceptiveFields_custom import get_receptive_fields
from utils_custom.get_db_attrs import get_db_attrs
from generate_db_CNNfeats import generate_db_CNNfeats

IMSCALE = 0.5
NUM_REF_IMAGE = 1175
IM_FIXED_H = 586
PAD_VAL = 255
NUM_CHANNELS = 3


def preprocess_im(img_path, num_img, scale, fixed_h, pad_val):
    ims = []
    trace_H, trace_W = 0, 0
    for i in range(1, num_img+1):
        im = cv2.imread(os.path.join(img_path, f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        # only 4 shoes are 1 pixel taller (all images -> height=586, only 4 images -> height=587)
        im = im[:fixed_h, :]

        if i in [107, 525]:
            im = cv2.flip(im, 1)

        im_H, im_W = im.shape
        pad_left = np.full((im_H, (270 - im_W) // 2), pad_val, dtype=np.uint8)
        pad_right = np.full((im_H, (270 - im_W + 1) // 2), pad_val, dtype=np.uint8)
        im_padded = np.hstack((pad_left, im, pad_right))
        im_pad_resized = cv2.resize(im_padded, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        if i == 1:
            trace_H, trace_W = im_pad_resized.shape
            
        ims.append(im_pad_resized)

    # zero-center the data
    # ims.shape = (batch_size=1175, height=293, width=135)
    ims = np.array(ims)
    mean_one_im = np.mean(ims, axis=0)
    mean_one_pix = np.mean(np.mean(mean_one_im, axis=0))
    ims_zero_centered = ims - mean_one_pix
    ims_4D = np.tile(ims_zero_centered[:, :, np.newaxis], (1, 1, NUM_CHANNELS, 1))

    return ims_4D, trace_H, trace_W


def save_feats_fid300(dbname, db_feats_info, save_combined=True):
    feats_path = os.path.join('feats', dbname)
    os.makedirs(feats_path, exist_ok=True)
    
    feats_gen_info = os.path.join(feats_path, 'fid300_feat_info.pkl')
    with open(feats_gen_info, 'wb') as file:
        pickle.dump({
            'feat_dims': db_feats_info['feat_dims'],
            'receptive_fields': db_feats_info['receptive_fields'],
            'trace_H': db_feats_info['trace_H'],
            'trace_W': db_feats_info['trace_W']
            }, file)

    # save all_db_feats
    if save_combined:
        feats_all_path = os.path.join(feats_path, 'fid300_all.pkl')
        
        if not os.path.exists(feats_all_path):
            with open(feats_all_path, 'wb') as file:
                pickle.dump(db_feats_info, file)
                
    # save each db_feats into separate file
    else:
        assert NUM_REF_IMAGE == db_feats_info['db_feats'].shape[0]
        for i in range(NUM_REF_IMAGE):
            db_feats = db_feats_info['db_feats'][i, :, :, :]
            db_labels = db_feats_info['db_labels'][:, :, :, i]
            feats_each_path = os.path.join(feats_path, f'fid300_{i+1:03d}.pkl')
            
            if not os.path.exists(feats_each_path):
                with open(feats_each_path, 'wb') as file:
                    pickle.dump({
                        'db_feats': db_feats,
                        'db_labels': db_labels
                    }, file)

    

def gen_feats_fid300(db_ind=2):
    db_attr, _, dbname = get_db_attrs('fid300', db_ind)

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
    treadids = np.zeros(NUM_REF_IMAGE)
    id = 0
    for g in groups:
        id += 1
        for p in g:
            treadids[p-1] = id
    
    for p in range(NUM_REF_IMAGE):
        if treadids[p] == 0:
            id += 1
            treadids[p] = id

    net = ResNet50Encoder(db_ind=2, db_attr=db_attr)
    
    ref_img_path = os.path.join('datasets', 'FID-300', 'references')
    ref_imgs_processed, trace_H, trace_W = preprocess_im(ref_img_path, \
        NUM_REF_IMAGE, IMSCALE, IM_FIXED_H, PAD_VAL)
    
    # ref_imgs_processed.shape = (batch_size=1175, height=293, width=135, channels=3)
    #NOTE: all_db_feats.shape = (batch_size=1175, out_channels=256, height=147, width=68)
    # ref_imgs_processed = )
    all_db_feats = generate_db_CNNfeats(net, np.transpose(ref_imgs_processed, (0, 2, 1, 3)))
    all_db_labels = treadids.reshape(1, 1, 1, -1)

    feat_dims = all_db_feats.shape
    rf = get_receptive_fields(net.model[0])
    
    db_feats_info = {
        'db_feats': all_db_feats,
        'db_labels': all_db_labels,
        'feat_dims': feat_dims,
        'receptive_fields': rf,
        'trace_H': trace_H,
        'trace_W': trace_W
    }
    save_feats_fid300(dbname, db_feats_info, save_combined=True)