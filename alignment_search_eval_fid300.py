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


def load_first_feat(dbname):
    '''pickle.load(f) returns a dictionary which has
    db_feats, db_labels, feat_dims, rfsIm, trace_H, trace_W as keys'''
    
    first_feat_path = os.path.join(os.path.join('feats', dbname), 'fid300_001.pkl')
    
    with open(first_feat_path, 'rb') as f:
        fid300_001 = pickle.load(f)
    
    return fid300_001



def fill_db_feats(db_feats_first, db_chunk_inds, dbname):
    #NOTE - Load the entire db_feats
    # if os.path.isfile(os.path.join('feats', dbname, 'fid300_all.pkl')):
    #     with open(os.path.join('feats', dbname, 'fid300_all.pkl'), 'rb') as f:
    #         dat = pickle.load(f)
    #     return dat['db_feats']
    
    len_db_chunks = db_chunk_inds[1] - db_chunk_inds[0]
    db_feats = np.zeros((db_feats_first.shape[0], db_feats_first.shape[1], 
                     db_feats_first.shape[2], len_db_chunks), dtype=db_feats_first.dtype)

    # Loading specified chunks of the database and filling the db_feats array
    start_idx, end_idx = db_chunk_inds
    for i, ind in enumerate(range(start_idx, end_idx)):
        filename = os.path.join('feats', dbname, f'fid300_{ind:03d}.pkl')
        with open(filename, 'rb') as filename:
            dat = pickle.load(filename)
        db_feats[:, :, :, i] = dat['db_feats']

    return db_feats



def preprocess_p_im(fname, imscale, trace_H, trace_W):
    p_im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    p_im_resized = cv2.resize(p_im, (0, 0), fx=imscale, fy=imscale)
    p_H, p_W = p_im_resized.shape
    
    # Fix latent prints are bigger than the test impressions
    if p_H > p_W and p_H > trace_H:
        p_im_resized = cv2.resize(p_im_resized, (trace_H, int((trace_H / p_H) * p_W)))
    elif p_W >= p_H and p_W > trace_W:
        p_im_resized = cv2.resize(p_im_resized, (int((trace_W / p_W) * p_H), trace_W))
        
    # Subtract mean_im_pix from p_im
    # p_im = p_im.astype(np.float32) - mean_im_pix
    mean_im_pix_dict = sio.loadmat(os.path.join('results', 'latent_ims_mean_pix.mat'))
    mean_im_pix = mean_im_pix_dict['mean_im_pix']
    mean_im_expanded = cv2.resize(mean_im_pix, (p_im_resized.shape[1], p_im_resized.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Since p_im is a single channel image, you might want to subtract each channel of mean_im_expanded from p_im separately
    p_im_resized_exp = np.expand_dims(p_im_resized, axis=2)
    p_im_processed = p_im_resized_exp - mean_im_expanded
    
    return p_im_processed



def pad_per_angle(center, p_idx, angle, p_im_padded, p_mask_padded):
    rows, cols, _ = p_im_padded.shape
    center = (cols / 2, rows / 2)
        
    # Creating rotation matrices
    rot_mat_im = cv2.getRotationMatrix2D(center, angle, 1)
    rot_mat_mask = cv2.getRotationMatrix2D(center, angle, 1)
    
    # Rotating images
    p_im_padded_r = cv2.warpAffine(p_im_padded, rot_mat_im, (cols, rows), \
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    # To prevent cv::UMat format error, we need to convert p_mask_padded to float32 numpy array
    p_mask_padded_r = cv2.warpAffine(np.float32(p_mask_padded), rot_mat_mask, (cols, rows), \
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return p_im_padded_r, p_mask_padded_r



def return_feat_ijr(p_r_feat, first_feat, h, w, offsety, offsetx, p_mask_padded_r, trace_H, trace_W, erode_pct, db_ind):
    feat_dims = first_feat['feat_dims']
    rfsIm = first_feat['rfsIm']
    trace_H = first_feat['trace_H']
    trace_W = first_feat['trace_W']
    
    im_f2i = feat_2_image(rfsIm)
    radius = max(1, np.floor(min(feat_dims[1], feat_dims[2]) * erode_pct))
    radius = int(radius)
    se = np.ones((radius, radius))
    
    pix_i = offsety + h * 4
    pix_j = offsetx + w * 4
    
    # p_r_feat.shape = (batch_size, channel_size, height, width)
    # feat_dims = (batch_size=1175, out_channel_size=256, height=147, width=68)
    p_ijr_feat = p_r_feat[:, :, h:h+feat_dims[2], w:w+feat_dims[3]]
    p_mask_ijr = p_mask_padded_r[pix_i:pix_i+trace_H, pix_j:pix_j+trace_W]
    
    p_ijr_feat_mask = warp_masks(p_mask_ijr, im_f2i, feat_dims, db_ind) # Placeholder
    p_ijr_feat_mask = cv2.copyMakeBorder(p_ijr_feat_mask, radius, radius, radius, radius, cv2.BORDER_CONSTANT, value=0)
    p_ijr_feat_mask = cv2.erode(p_ijr_feat_mask, se)
    #FIXME - IndexError(should be 3D, but 2D array is given)
    p_ijr_feat_mask = p_ijr_feat_mask[radius:-radius, radius:-radius]
    
    # p_ijr_feat.shape, p_ijr_feat_mask.shape = (torch.Size([1, 256, 147, 68]), (293, 135))
    return p_ijr_feat, p_ijr_feat_mask



def print_msg(cnt, angles, eraseStr, pad_H, pad_W):
    msg = f'{cnt}/{len(angles) * np.ceil((pad_H/2)+0.5) * np.ceil((pad_W/2)+0.5)}'
    if cnt % 10 == 0:
        print(eraseStr + msg, end='')
        eraseStr = '\b' * len(msg)
    return eraseStr



def alignment_search_eval_fid300(p_inds, db_ind=2):
    IMSCALE = 0.5
    ERODE_PCT = 0.1
    
    # Define initial variables
    db_attr, db_chunks, dbname = get_db_attrs('fid300', db_ind)
    # len(db_chunks) is 1175
    db_chunk_inds = db_chunks[0]
    
    #FIXME - For debugging, use only 100 chunks
    db_chunk_inds = (1, 101)
    
    net = ModifiedNetwork(db_ind=2, db_attr=db_attr)
    first_feat = load_first_feat(dbname)
    
    db_feats_first = first_feat['db_feats']
    db_feats = fill_db_feats(db_feats_first, db_chunk_inds, dbname)
    
    feat_dims = first_feat['feat_dims']
    rfsIm = first_feat['rfsIm']
    trace_H = first_feat['trace_H']
    trace_W = first_feat['trace_W']

    # feat_dims[1] = 256 (out_channel_size)
    ones_w = torch.ones((feat_dims[1], 1, 1), dtype=torch.float32).cuda()
    
    # First, 'results/<dbnmae>' path needs to be created
    if not os.path.exists(os.path.join('results', dbname)):
        os.makedirs(os.path.join('results', dbname), exist_ok=True)
    
    # p_inds = [start, end]
    for p in range(p_inds[0], p_inds[1]+1):
        fname = os.path.join('results', dbname, f'fid300_alignment_search_ones_res_{p:04d}.mat')
        
        # if os.path.exists(fname):
        #     continue
        # lock_fname = fname + '.lock'
        # if os.path.exists(lock_fname):
        #     continue
        
        # if the file does not exist, a+ option creates it ('a' option assumes the file exists)
        # fid = open(lock_fname, 'a+')
        # fid.write(f'p={time.time()}')
        
        # Read and resize the image
        p_im_fname = os.path.join('datasets', 'FID-300', 'tracks_cropped', f'{p:05d}.jpg')
        p_im = preprocess_p_im(p_im_fname, IMSCALE, trace_H, trace_W)
        
        # Pad the latent print
        pad_H = trace_H - p_im.shape[0]
        pad_W = trace_W - p_im.shape[1]
        assert pad_H >= 0 and pad_W >= 0, f'pad_H={pad_H}, pad_W={pad_W}'
        
        # Padding: p_im.shape = (H, W, 3) -> 3D. In MATLAB code, it is 2D.
        p_im_padded = np.pad(p_im, ((pad_H, pad_H), (pad_W, pad_W), (0,0)), \
            mode='constant', constant_values=255)
        #NOTE - do not remove the code below -> it is used later
        p_mask_padded = np.pad(np.ones(p_im.shape, dtype=bool), ((pad_H, pad_H), (pad_W, pad_W), \
            (0, 0)), mode='constant', constant_values=0)
        
        cnt = 0
        eraseStr = ''

        angles = np.arange(-20, 21, 4)  # Creating an array from -20 to 20 with a step of 4
        num_angles = len(angles)
        transx = np.arange(1, pad_W+2, 2)  # Creating an array from 1 to pad_W+1 with a step of 2
        transy = np.arange(1, pad_H+2, 2)  # Creating an array from 1 to pad_H+1 with a step of 2

        # Initializing scores_ones with zeros
        scores_ones = np.zeros((len(transy), len(transx), num_angles, db_feats.shape[3]), dtype=np.float32)

        rows, cols, _ = p_im_padded.shape
        center = (cols / 2, rows / 2)
        
        for ang_idx in range(num_angles):
            ang = angles[ang_idx]
            #NOTE - Use this function for real tests
            # p_im_padded_r, p_mask_padded_r = pad_per_angle(center, p, r, p_im_padded, p_mask_padded)
            
            # Just load images created in MATLAB code for test (tentative approach)
            p_im_padded_r = cv2.imread(os.path.join('results', 'resnet_4x_matlab', \
                f'fid300_rotated_im_{p:04d}_{ang:03d}.jpg'), cv2.IMREAD_COLOR)
            p_mask_padded_r = cv2.imread(os.path.join('results', 'resnet_4x_matlab', \
                f'fid300_rotated_mask_{p:04d}_{ang:03d}.jpg'), cv2.IMREAD_GRAYSCALE)
                    
            offsets_y = [0]
            if pad_H > 1:
                offsets_y.append(2)
            
            offsets_x = [0]
            if pad_W > 1:
                offsets_x.append(2)

            for offsetx in offsets_x:
                for offsety in offsets_y:
                    p_r_feat = generate_db_CNNfeats_gpu(net, p_im_padded_r[offsety:, offsetx:, :])
                    
                    h_margin = p_r_feat.shape[3] - feat_dims[3]
                    w_margin = p_r_feat.shape[2] - feat_dims[2]

                    for h in range(h_margin+1):
                        for w in range(w_margin+1):
                            eraseStr = print_msg(cnt, angles, eraseStr, pad_H, pad_W)
                            
                            pix_i = offsety + h * 4
                            pix_j = offsetx + w * 4
                            
                            if pix_i + trace_H > p_mask_padded_r.shape[0] or \
                            pix_j + trace_W > p_mask_padded_r.shape[1]:
                                continue
                            
                            # The next operations are placeholders and need actual Python functions
                            p_ijr_feat, p_ijr_feat_mask = return_feat_ijr(p_r_feat, first_feat, h, w, \
                                offsety, offsetx, p_mask_padded_r, trace_H, trace_W, ERODE_PCT, db_ind)
                            
                            p_ijr_feat_mask = torch.tensor(p_ijr_feat_mask, dtype=torch.float32).to('cuda')
                            db_feats = torch.tensor(db_feats, dtype=torch.float32).to('cuda')
                            
                            #REVIEW: p_ijr_feat.shape = torch.Size([1, 147, 217, 84]) -> fixed to torch.Size([256, 147, 68])
                            p_ijr_feat = p_ijr_feat.squeeze(0)
                            # scores_cell.shape = torch.Size([1, 1, 1, 100])
                            scores_cell = weighted_masked_NCC_features(db_feats, p_ijr_feat, p_ijr_feat_mask, ones_w)  # Placeholder
                            scores_ones[int(pix_i/2+0.5), int(pix_j/2+0.5), ang_idx, :] = scores_cell[0]
                            cnt += 1
            
        minsONES = np.max(np.max(np.max(scores_ones, axis=0), axis=0), axis=0)
        locaONES = scores_ones == minsONES
        np.savez(fname, scores_ones, minsONES, locaONES)

        # fid.close()
        # os.remove(lock_fname)
