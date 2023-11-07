import cv2
import numpy as np
import os
import pickle
import scipy.io as sio

def rotate_img_mask(angle, q_im_padded, q_mask_padded):
    _, rows, cols = q_im_padded.shape
    center = (cols / 2, rows / 2)
        
    # Creating rotation matrices
    rot_mat_im = cv2.getRotationMatrix2D(center, angle, 1)
    rot_mat_mask = cv2.getRotationMatrix2D(center, angle, 1)
    
    # Rotating images
    q_im_padded_ang = cv2.warpAffine(q_im_padded, rot_mat_im, (cols, rows), \
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    # To prevent cv::UMat format error, we need to convert p_mask_padded to float32 numpy array
    q_mask_padded_ang = cv2.warpAffine(np.float32(q_mask_padded), rot_mat_mask, (cols, rows), \
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return q_im_padded_ang, q_mask_padded_ang


def pad_img_mask(q_im, pad_H, pad_W):
    # Padding: q_im.shape = (H, W, 3) -> 3D. In MATLAB code, it is 2D
    q_im_padded = np.pad(q_im, ((pad_H, pad_H), (pad_W, pad_W), (0,0)), \
    mode='constant', constant_values=255)
    q_mask_padded = np.pad(np.ones(q_im.shape, dtype=bool), ((pad_H, pad_H), (pad_W, pad_W), \
    (0, 0)), mode='constant', constant_values=0)
    
    return q_im_padded, q_mask_padded


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


# Create an array of angles from -20 to 20 with a step of 4
angles = np.arange(-20, 21, 4)  
num_angles = len(angles)
query_ind = (1, 301)
IMSCALE = 0.5

dbname = 'resnet_4x'
new_dbname = 'resnet_4x_python'

feats_info_path  = os.path.join('feats', dbname, 'fid300_feat_info.pkl')
with open(feats_info_path, 'rb') as f:
    feats_gen_info = pickle.load(f)
trace_H, trace_W = feats_gen_info['trace_H'], feats_gen_info['trace_W']

for qidx in range(query_ind[0], query_ind[1]+1):
    query_im_fname = os.path.join('datasets', 'FID-300', 'tracks_cropped', f'{qidx:05d}.jpg')
    q_im = preprocess_query_im(query_im_fname, IMSCALE, trace_H, trace_W)

    pad_H = trace_H - q_im.shape[0]
    pad_W = trace_W - q_im.shape[1]

    q_im_padded, q_mask_padded = pad_img_mask(q_im, pad_H, pad_W)

    qidx_save_dir = os.path.join('results', new_dbname, f'{qidx:04d}')
    if not os.path.exists(qidx_save_dir):
        os.mkdir(qidx_save_dir)
    
    
    for ang_idx in range(num_angles):
        ang = angles[ang_idx]
        q_im_padded_ang, q_mask_padded_ang = rotate_img_mask(ang, q_im_padded, q_mask_padded)
        
        cv2.imwrite(os.path.join(qidx_save_dir, f'fid300_rotated_im_{qidx:04d}_{ang:03d}.jpg'), q_im_padded_ang)
        cv2.imwrite(os.path.join(qidx_save_dir, f'fid300_rotated_im_{qidx:04d}_{ang:03d}.jpg'), q_mask_padded_ang)