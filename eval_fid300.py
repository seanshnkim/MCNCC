import torch
import cv2
import numpy as np
import scipy.io as sio
import os
import pickle
import time
import logging

from utils_custom.get_db_attrs import get_db_attrs
from utils_custom.weighted_masked_NCC_features_no_align import weighted_masked_NCC_features
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



def pad_img_mask(q_im, pad_H, pad_W):
    # Padding: q_im.shape = (H, W, 3) -> 3D. In MATLAB code, it is 2D
    q_im_padded = np.pad(q_im, ((pad_H, pad_H), (pad_W, pad_W), (0,0)), \
    mode='constant', constant_values=255)
    q_mask_padded = np.pad(np.ones(q_im.shape, dtype=bool), ((pad_H, pad_H), (pad_W, pad_W), \
    (0, 0)), mode='constant', constant_values=0)
    
    return q_im_padded, q_mask_padded



def eval_fid300(query_ind, db_ind=2):
    IMSCALE = 0.5
    ERODE_PCT = 0.1
    
    db_attr, db_chunks, dbname = get_db_attrs('fid300', db_ind)
    # db_chunk_inds = [1, 1176]. 1175 is equal to the number of reference images in FID-300 dataset.
    #FIXME - Since it takes too long to test with entire data, use only 100 chunks
    #NOTE: This is original code: 
    db_chunk_inds = db_chunks[0]
    db_chunk_inds = (1, 101)
    
    net = ResNet50Encoder(db_ind=2, db_attr=db_attr)
    
    feats_info_path  = os.path.join('feats', dbname, 'fid300_feat_info.pkl')
    with open(feats_info_path, 'rb') as f:
        feats_gen_info = pickle.load(f)
    feat_dims = feats_gen_info['feat_dims']
    trace_H, trace_W = feats_gen_info['trace_H'], feats_gen_info['trace_W']
    data_type = feats_gen_info['data_type']
    
    num_feats, feat_out_ch, feat_H, feat_W = feat_dims
    num_k, ksize = 1, 1
    weight_ones = torch.ones((num_k, feat_out_ch, ksize, ksize), dtype=torch.float32).cuda()
    
    # db_feats.shape = (1175, 256, 147, 68)
    # If load_combined=True, load all 1175 reference images.
    db_chunk_feats = load_db_chunk_feats(feat_dims, data_type, db_chunk_inds, dbname)
    db_chunk_feats = torch.tensor(db_chunk_feats, dtype=torch.float32).to('cuda')
    
    new_dbname = 'resnet_4x_no_align'
    if not os.path.exists(os.path.join('results', new_dbname)):
        os.makedirs(os.path.join('results', new_dbname), exist_ok=True)
    
    logging.basicConfig(filename=os.path.join('results', new_dbname, 'fid300_ones_res.log'), \
        level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    
    start_time = time.time()
    for qidx in range(query_ind[0], query_ind[1]+1):
        score_save_fname = os.path.join('results', new_dbname, \
            f'fid300_ones_res_{qidx:04d}.npz')
        
        query_im_fname = os.path.join('datasets', 'FID-300', 'tracks_cropped', f'{qidx:05d}.jpg')
        q_im = preprocess_query_im(query_im_fname, IMSCALE, trace_H, trace_W)
        
        pad_H = trace_H - q_im.shape[0]
        pad_W = trace_W - q_im.shape[1]
        assert pad_H >= 0 and pad_W >= 0, f'pad_H={pad_H}, pad_W={pad_W}'

        q_im_padded, q_mask_padded = pad_img_mask(q_im, pad_H, pad_W)
        query_feat = generate_db_CNNfeats_gpu(net, q_im_padded)
        q_mask_padded = torch.tensor(q_mask_padded, dtype=torch.float32).to('cuda')
        
        scores_ones = weighted_masked_NCC_features(db_chunk_feats, query_feat, q_mask_padded, weight_ones)  # Placeholder
            
        minsONES = np.max(np.max(np.max(scores_ones, axis=1, keepdims=True), axis=2, keepdims=True), axis=3, keepdims=True)
        locaONES = scores_ones == minsONES
        np.savez(score_save_fname, scores=scores_ones, mins_ones=minsONES, loca_ones=locaONES)
        
    end_time = time.time()
    # Change it into hour, min, sec format
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
    logging.info(f"Query {qidx:03d} image took {elapsed_time} seconds.")