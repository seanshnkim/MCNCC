import torch
import numpy as np
from scipy.ndimage import binary_erosion, rotate
from scipy.io import loadmat, savemat
from skimage.transform import resize
import os
from utils_custom.get_db_attrs import get_db_attrs 
import pickle

from modified_network import ModifiedNetwork

def alignment_search_eval_fid300(p_inds, db_ind=2):
    imscale = 0.5
    erode_pct = 0.1

    db_attr, db_chunks, dbname = get_db_attrs('fid300', db_ind)

    net = ModifiedNetwork(db_ind=2, db_attr=db_attr)

    # mean_im_pix = loadmat(os.path.join('results', 'latent_ims_mean_pix.mat'))['mean_im_pix']
    # Load pickle file
    save_dir = os.path.join('feats', dbname)
    file_path = os.path.join(save_dir, 'fid300_001.pkl')
    
    with open(file_path 'rb') as f:
        mean_im_pix = pickle.load(f)

    # ... (loading and processing db_feats as in your MATLAB code)

    radius = max(1, np.floor(min(feat_dims[1], feat_dims[2]) * erode_pct))
    se = np.ones((radius, radius))

    ones_w = torch.ones((1, 1, feat_dims[3]), dtype=torch.float32).cuda()

    # db_feats to gpu, etc. 
    
    for p in np.reshape(p_inds, -1):
        fname = os.path.join('results', dbname, f'fid300_alignment_search_ones_res_{p:04d}.mat')
        if os.path.exists(fname):
            continue
        # ... (your lock file handling code)

        p_im = resize(imread(os.path.join('datasets', 'FID-300', 'tracks_cropped', f'{p:05d}.jpg')), imscale)
        # ... (resizing, padding, and other operations as in your MATLAB code)

        for r in angles:
            p_im_padded_r = rotate(p_im_padded, r, mode='constant', reshape=False)
            p_mask_padded_r = rotate(p_mask_padded, r, mode='constant', reshape=False, order=0)

            # ... (further processing and computations)

            save_results(fname, {'scores_ones': scores_ones, 'minsONES': minsONES, 'locaONES': locaONES})

# Some additional functions might need to be translated or imported, such as:
# - get_db_attrs
# - load_and_modify_network
# - generate_db_CNNfeats_gpu
# - weighted_masked_NCC_features
# - warp_masks
# - save_results
