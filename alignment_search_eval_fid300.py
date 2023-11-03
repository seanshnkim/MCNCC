import torch
import cv2
import numpy as np
import scipy.io as sio
import os
import pickle

from utils_custom.get_db_attrs import get_db_attrs
from utils_custom.warp_masks import warp_masks
from utils_custom.weighted_masked_NCC_features import weighted_masked_NCC_features
from utils_custom.feat_2_image import feat_2_image
from modified_network import ResNet50Encoder
from generate_db_CNNfeats_gpu import generate_db_CNNfeats_gpu


def load_db_chunk_feats(feat_dims, data_type, chunk_inds, dbname, load_combined=False):
    # NOTE - Load the entire db_feats
    if load_combined:
        with open(os.path.join('feats', dbname, 'fid300_all.pkl'), 'rb') as f:
            dat = pickle.load(f)
        return dat['db_feats']
    
    num_feats, feat_out_ch, feat_H, feat_W = feat_dims
    chunk_size = chunk_inds[1] - chunk_inds[0]
    chunk_feats = np.zeros((chunk_size, feat_out_ch, feat_H, feat_W), dtype=data_type)

    # Load each feature vector separately and fill the db_feats array
    start_idx, end_idx = chunk_inds
    for ind in range(start_idx, end_idx):
        filename = os.path.join('feats', dbname, f'fid300_{ind:03d}.pkl')
        with open(filename, 'rb') as filename:
            dat = pickle.load(filename)
        chunk_feats[ind-1, :, :, :] = dat['db_feats']

    return chunk_feats



def preprocess_query_im(fname, imscale, trace_H, trace_W):
    query_im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    query_im_resized = cv2.resize(query_im, (0, 0), fx=imscale, fy=imscale)
    height, width = query_im_resized.shape
    
    # Fix latent prints are bigger than the test impressions
    if height > width and height > trace_H:
        query_im_resized = cv2.resize(query_im_resized, (trace_H, int((trace_H / height) * width)))
    elif width >= height and width > trace_W:
        query_im_resized = cv2.resize(query_im_resized, (int((trace_W / width) * height), trace_W))
    
    height, width = query_im_resized.shape
    # Subtract mean_im_pix from p_im
    mean_im_pix_dict = sio.loadmat(os.path.join('results', 'latent_ims_mean_pix.mat'))
    mean_im_pix = mean_im_pix_dict['mean_im_pix']
    
    # mean_im, query_im expand to 3D arrya:(result shape: (152, 102, 3))
    query_im_expanded = np.expand_dims(query_im_resized, axis=2)
    mean_im_expanded = cv2.resize(mean_im_pix, (width, height), interpolation=cv2.INTER_CUBIC)
    query_im_zero_mean = query_im_expanded - mean_im_expanded
    
    return query_im_zero_mean



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



def process_feat(q_feat, feat_info, h, w, offsety, offsetx, q_mask_padded, erode_pct, db_ind):
    feat_dims = feat_info['feat_dims']
    rfsIm = feat_info['receptive_fields']
    trace_H = feat_info['trace_H']
    trace_W = feat_info['trace_W']
    feat_H, feat_W = feat_dims[2], feat_dims[3]
    
    im_f2i = feat_2_image(rfsIm)
    radius = int(max(1, np.floor(min(feat_H, feat_W) * erode_pct)))
    se = np.ones((radius, radius))
    
    pix_i = offsety + h * 4
    pix_j = offsetx + w * 4
    
    # feat_dims = (batch_size=1175, out_channel_size=256, height=147, width=68)
    q_mask_pix = q_mask_padded[pix_i:pix_i+trace_H, pix_j:pix_j+trace_W]
    q_mask_pix = warp_masks(q_mask_pix, im_f2i, feat_dims, db_ind) # Placeholder
    q_mask_pix = cv2.copyMakeBorder(q_mask_pix, radius, radius, radius, radius, cv2.BORDER_CONSTANT, value=0)
    q_mask_pix = cv2.erode(q_mask_pix, se)
    q_mask_pix = q_mask_pix[radius:-radius, radius:-radius]
    
    #REVIEW: torch.Size([1, 147, 217, 84]) -> squeeze to torch.Size([out_ch=256, height=147, width=68])
    q_feat_hw = q_feat[:, :, h:h+feat_H, w:w+feat_W].squeeze(0)
    
    return q_feat_hw, q_mask_pix



def print_msg(cnt, angles, eraseStr, pad_H, pad_W):
    msg = f'{cnt}/{len(angles) * np.ceil((pad_H/2)+0.5) * np.ceil((pad_W/2)+0.5)}'
    if cnt % 10 == 0:
        print(eraseStr + msg, end='')
        eraseStr = '\b' * len(msg)
    return eraseStr



def alignment_search_eval_fid300(query_ind, db_ind=2):
    IMSCALE = 0.5
    ERODE_PCT = 0.1
    
    db_attr, db_chunks, dbname = get_db_attrs('fid300', db_ind)
    # db_chunk_inds = [1, 1176]. 1175 is equal to the number of reference images in FID-300 dataset.
    #FIXME - Since it takes too long to test with entire data, use only 100 chunks
    #NOTE: This is original code: db_chunk_inds = db_chunks[0]
    db_chunk_inds = (1, 101)
    
    net = ResNet50Encoder(db_ind=2, db_attr=db_attr)
    
    feats_info_path  = os.path.join('feats', dbname, 'fid300_feat_info.pkl')
    with open(feats_info_path, 'rb') as f:
        feats_gen_info = pickle.load(f)
    feat_dims = feats_gen_info['feat_dims']
    trace_H, trace_W = feats_gen_info['trace_H'], feats_gen_info['trace_W']
    data_type = feats_gen_info['data_type']
    
    # Create an array of angles from -20 to 20 with a step of 4
    angles = np.arange(-20, 21, 4)  
    num_angles = len(angles)
    num_feats, feat_out_ch, feat_H, feat_W = feat_dims
    num_k, ksize = 1, 1
    weight_ones = torch.ones((num_k, feat_out_ch, ksize, ksize), dtype=torch.float32).cuda()
    
    # db_feats.shape = (1175, 256, 147, 68)
    db_chunk_feats = load_db_chunk_feats(feat_dims, data_type, db_chunk_inds, dbname)
    db_chunk_feats = torch.tensor(db_chunk_feats, dtype=torch.float32).to('cuda')
    num_db_chunks = db_chunk_feats.shape[0]
    
    if not os.path.exists(os.path.join('results', dbname)):
        os.makedirs(os.path.join('results', dbname), exist_ok=True)
        
    for qidx in range(query_ind[0], query_ind[1]+1):
        score_save_fname = os.path.join('results', dbname, f'fid300_alignment_search_ones_res_{qidx:04d}.mat')
        # if os.path.exists(score_save_fname):
        #     continue
        # lock_fname = score_save_fname + '.lock'
        # if os.path.exists(lock_fname):
        #     continue
        
        # if the file does not exist, a+ option creates it ('a' option assumes the file exists)
        # fid = open(lock_fname, 'a+')
        # fid.write(f'p={time.time()}')
        
        query_im_fname = os.path.join('datasets', 'FID-300', 'tracks_cropped', f'{qidx:05d}.jpg')
        q_im = preprocess_query_im(query_im_fname, IMSCALE, trace_H, trace_W)
        
        pad_H = trace_H - q_im.shape[0]
        pad_W = trace_W - q_im.shape[1]
        assert pad_H >= 0 and pad_W >= 0, f'pad_H={pad_H}, pad_W={pad_W}'
        
        # Padding: q_im.shape = (H, W, 3) -> 3D. In MATLAB code, it is 2D.
        #NOTE - do not remove the code below -> it is used later
        # p_im_padded = np.pad(q_im, ((pad_H, pad_H), (pad_W, pad_W), (0,0)), \
        #     mode='constant', constant_values=255)
        # p_mask_padded = np.pad(np.ones(q_im.shape, dtype=bool), ((pad_H, pad_H), (pad_W, pad_W), \
        #     (0, 0)), mode='constant', constant_values=0)
        
        cnt = 0
        eraseStr = ''
        transx = np.arange(1, pad_W+2, 2)  # Creating an array from 1 to pad_W+1 with a step of 2
        transy = np.arange(1, pad_H+2, 2)
        # Initialize scores_ones with zeros
        scores_ones = np.zeros((num_db_chunks, len(transy), len(transx), num_angles), dtype=np.float32)
        
        for ang_idx in range(num_angles):
            ang = angles[ang_idx]
            #NOTE - Use this function for real tests
            # p_im_padded_r, p_mask_padded_r = pad_per_angle(center, p, r, p_im_padded, p_mask_padded)
            
            # Just load images created in MATLAB code for test (tentative approach)
            q_im_padded_ang = cv2.imread(os.path.join('results', 'resnet_4x_matlab', \
                f'fid300_rotated_im_{qidx:04d}_{ang:03d}.jpg'), cv2.IMREAD_COLOR)
            q_mask_padded_ang = cv2.imread(os.path.join('results', 'resnet_4x_matlab', \
                f'fid300_rotated_mask_{qidx:04d}_{ang:03d}.jpg'), cv2.IMREAD_GRAYSCALE)
            q_mask_padded_H, q_mask_padded_W = q_mask_padded_ang.shape
                    
            offsets_y = [0]
            if pad_H > 1:
                offsets_y.append(2)
            
            offsets_x = [0]
            if pad_W > 1:
                offsets_x.append(2)

            for offsetx in offsets_x:
                for offsety in offsets_y:
                    # query_feat_off.shape = torch.Size([1, 256, 217, 84])
                    query_feat_off = generate_db_CNNfeats_gpu(net, q_im_padded_ang[offsety:, offsetx:, :])
                    qH, qW = query_feat_off.shape[2], query_feat_off.shape[3]
                    h_margin = qH - feat_H
                    w_margin = qW - feat_W

                    for h in range(h_margin+1):
                        for w in range(w_margin+1):
                            eraseStr = print_msg(cnt, angles, eraseStr, pad_H, pad_W)
                            
                            pix_i = offsety + h * 4
                            pix_j = offsetx + w * 4
                            
                            if pix_i + trace_H > q_mask_padded_H or \
                            pix_j + trace_W > q_mask_padded_W:
                                continue
                            
                            # The next operations are placeholders and need actual Python functions
                            query_feat_hw, query_feat_mask_hw = process_feat(query_feat_off, feats_gen_info, h, w, \
                                offsety, offsetx, q_mask_padded_ang, ERODE_PCT, db_ind)
                            
                            query_feat_mask_hw = torch.tensor(query_feat_mask_hw, dtype=torch.float32).to('cuda')
                            
                            # scores_cell.shape = torch.Size([100, 1, 1, 1])
                            scores_cell = weighted_masked_NCC_features(db_chunk_feats, query_feat_hw, query_feat_mask_hw, weight_ones)  # Placeholder
                            # scores_ones.shape = (100, 71, 17, 11)
                            scores_ones[:, int(pix_i/2+0.5), int(pix_j/2+0.5), ang_idx] = scores_cell.squeeze()
                            cnt += 1
            
        minsONES = np.max(np.max(np.max(scores_ones, axis=1, keepdims=True), axis=2, keepdims=True), axis=3, keepdims=True)
        locaONES = scores_ones == minsONES
        np.savez(score_save_fname, scores_ones, minsONES, locaONES)

        # fid.close()
        # os.remove(lock_fname)