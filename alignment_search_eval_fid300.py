import torch
import cv2
import numpy as np
# from scipy.io import loadmat, savemat
import scipy.io as sio
import os
import pickle
import time

from utils_custom.get_db_attrs import get_db_attrs
from utils_custom.warp_masks import warp_masks
from utils_custom.weighted_masked_NCC_features import weighted_masked_NCC_features
from utils_custom.feat_2_image import feat_2_image
from modified_network import ModifiedNetwork
from generate_db_CNNfeats_gpu import generate_db_CNNfeats_gpu

def alignment_search_eval_fid300(p_inds, db_ind=2):
    imscale = 0.5
    erode_pct = 0.1

    db_attr, db_chunks, dbname = get_db_attrs('fid300', db_ind)

    # load and modify network
    net = ModifiedNetwork(db_ind=2, db_attr=db_attr)

    mean_im_pix_dict = sio.loadmat(os.path.join('results', 'latent_ims_mean_pix.mat'))
    mean_im_pix = mean_im_pix_dict['mean_im_pix']
    
    # load database chunk
    db_save_dir = os.path.join('feats', dbname)
    first_feat_path = os.path.join(db_save_dir, 'fid300_001.pkl')
    
    # pickle.load(f) returns a dictionary which has
    # db_feats, db_labels, feat_dims, rfsIm, trace_H, trace_W as keys
    with open(first_feat_path, 'rb') as f:
        fid300_001 = pickle.load(f)
        db_feats_init = fid300_001['db_feats']
        feat_dims = fid300_001['feat_dims']
        rfsIm = fid300_001['rfsIm']
        trace_H = fid300_001['trace_H']
        trace_W = fid300_001['trace_W']
    
    # len(db_chunks) is 1175
    db_chunk_inds = db_chunks[0]
    # db_feats.shape = (1175, 256, 2, 1175)
    db_feats = np.zeros((db_feats_init.shape[0], db_feats_init.shape[1], 
                     db_feats_init.shape[2], len(db_chunk_inds)), dtype=db_feats_init.dtype)

    # Loading specified chunks of the database and filling the db_feats array
    for i, ind in enumerate(db_chunk_inds):
        filename = os.path.join('feats', dbname, f'fid300_{ind:03d}.pkl')
        with open(filename, 'rb') as filename:
            dat = pickle.load(filename)
        db_feats[:, :, :, i] = dat['db_feats']

    im_f2i = feat_2_image(rfsIm)

    radius = max(1, np.floor(min(feat_dims[1], feat_dims[2]) * erode_pct))
    se = np.ones((radius, radius))

    ones_w = torch.ones((1, 1, feat_dims[3]), dtype=torch.float32).cuda()
    
    # First, 'results/<dbnmae>' path needs to be created
    if not os.path.exists(os.path.join('results', dbname)):
        os.makedirs(os.path.join('results', dbname), exist_ok=True)
    
    # p_inds = [start, end]
    for p in range(p_inds[0], p_inds[1]+1):
        fname = os.path.join('results', dbname, f'fid300_alignment_search_ones_res_{p:04d}.mat')
        if os.path.exists(fname):
            continue
        lock_fname = fname + '.lock'
        if os.path.exists(lock_fname):
            continue
        
        # if the file does not exist, a+ option creates it ('a' option assumes the file exists)
        fid = open(lock_fname, 'a+')
        fid.write(f'p={time.time()}')
        
        # Read and resize the image
        # We need 2D dimension numpy array for p_im (in MATLAB code)
        p_im = cv2.imread(os.path.join('datasets', 'FID-300', 'tracks_cropped', f'{p:05d}.jpg'), cv2.IMREAD_GRAYSCALE)
        p_im = cv2.resize(p_im, (0, 0), fx=imscale, fy=imscale)
        p_H, p_W = p_im.shape

        # Fix latent prints are bigger than the test impressions
        if p_H > p_W and p_H > trace_H:
            p_im = cv2.resize(p_im, (trace_H, int((trace_H / p_H) * p_W)))
        elif p_W >= p_H and p_W > trace_W:
            p_im = cv2.resize(p_im, (int((trace_W / p_W) * p_H), trace_W))
        
        # Subtract mean_im_pix from p_im
        # p_im = p_im.astype(np.float32) - mean_im_pix
        mean_im_expanded = cv2.resize(mean_im_pix, (p_im.shape[1], p_im.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Since p_im is a single channel image, you might want to subtract each channel of mean_im_expanded from p_im separately
        p_im_ch1 = p_im - mean_im_expanded[:, :, 0]
        p_im_ch2 = p_im - mean_im_expanded[:, :, 1]
        p_im_ch3 = p_im - mean_im_expanded[:, :, 2]
        p_im = np.stack((p_im_ch1, p_im_ch2, p_im_ch3), axis=2)
        p_H, p_W, p_C = p_im.shape
        
        # Pad the latent print
        pad_H = trace_H - p_H
        pad_W = trace_W - p_W
        
        # Padding p_im and a logical ones matrix
        # p_im.shape = (H, W, 3) -> 3D. In MATLAB code, it is 2D.
        p_im_padded = np.pad(p_im, ((pad_H, pad_H), (pad_W, pad_W), (0,0)), \
            mode='constant', constant_values=255)
        p_mask_padded = np.pad(np.ones((p_H, p_W, p_C), dtype=bool), ((pad_H, pad_H), (pad_W, pad_W), \
            (0, 0)), mode='constant', constant_values=0)
        
        cnt = 0
        eraseStr = ''

        angles = np.arange(-20, 21, 4)  # Creating an array from -20 to 20 with a step of 4
        transx = np.arange(1, pad_W+2, 2)  # Creating an array from 1 to pad_W+1 with a step of 2
        transy = np.arange(1, pad_H+2, 2)  # Creating an array from 1 to pad_H+1 with a step of 2

        # Initializing scores_ones with zeros
        scores_ones = np.zeros((len(transy), len(transx), len(angles), db_feats.shape[3]), dtype=np.float32)

        rows, cols, _ = p_im_padded.shape
        center = (cols / 2, rows / 2)

        for r in angles:
            # Creating rotation matrices
            rot_mat_im = cv2.getRotationMatrix2D(center, r, 1)
            rot_mat_mask = cv2.getRotationMatrix2D(center, r, 1)
            
            # Rotating images
            p_im_padded_r = cv2.warpAffine(p_im_padded, rot_mat_im, (cols, rows), \
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            # To prevent cv::UMat format error, we need to convert p_mask_padded to float32 numpy array
            p_mask_padded_r = cv2.warpAffine(np.float32(p_mask_padded), rot_mat_mask, (cols, rows), \
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    
            offsets_y = [0]
            if pad_H > 1:
                offsets_y.append(2)
            
            offsets_x = [0]
            if pad_W > 1:
                offsets_x.append(2)
            
            for offsetx in offsets_x:
                for offsety in offsets_y:
                    p_r_feat = generate_db_CNNfeats_gpu(net, p_im_padded_r[offsety:, offsetx:, :])
                    
                    for j in range(p_r_feat.shape[1] - feat_dims[2] + 1):
                        for i in range(p_r_feat.shape[0] - feat_dims[1] + 1):
                            
                            msg = f'{cnt}/{len(angles) * np.ceil((pad_H/2)+0.5) * np.ceil((pad_W/2)+0.5)} '
                            if cnt % 10 == 0:
                                print(eraseStr + msg, end='')
                                eraseStr = '\b' * len(msg)
                            
                            pix_i = offsety + (i - 1) * 4
                            pix_j = offsetx + (j - 1) * 4
                            
                            if pix_i + trace_H > p_mask_padded_r.shape[0] or \
                            pix_j + trace_W > p_mask_padded_r.shape[1]:
                                continue
                            
                            # The next operations are placeholders and need actual Python functions
                            p_ijr_feat = p_r_feat[i:i+feat_dims[1], j:j+feat_dims[2], :]
                            p_mask_ijr = p_mask_padded_r[pix_i:pix_i+trace_H, pix_j:pix_j+trace_W]
                            
                            p_ijr_feat_mask = warp_masks(p_mask_ijr, im_f2i, feat_dims, db_ind) # Placeholder
                            p_ijr_feat_mask = cv2.copyMakeBorder(p_ijr_feat_mask, radius, radius, radius, radius, cv2.BORDER_CONSTANT, value=0)
                            p_ijr_feat_mask = cv2.erode(p_ijr_feat_mask, se)
                            p_ijr_feat_mask = p_ijr_feat_mask[radius:-radius, radius:-radius, :]
                            
                            scores_cell = weighted_masked_NCC_features(db_feats, p_ijr_feat, p_ijr_feat_mask, ones_w)  # Placeholder
                            
                            scores_ones[int(pix_i/2+0.5), int(pix_j/2+0.5), r, :] = scores_cell[0]
                            cnt += 1
            
        minsONES = np.max(np.max(np.max(scores_ones, axis=0), axis=0), axis=0)
        locaONES = scores_ones == minsONES
        np.savez(fname, scores_ones, minsONES, locaONES)

        fid.close()
        os.remove(lock_fname)

# Some additional functions might need to be translated or imported, such as:
# - get_db_attrs
# - load_and_modify_network
# - generate_db_CNNfeats_gpu
# - weighted_masked_NCC_features
# - warp_masks
# - save_results
