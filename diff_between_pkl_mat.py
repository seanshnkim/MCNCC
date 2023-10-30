import pickle
import os
# import scipy.io as sio
import mat73

feats_dir = os.path.join('feats', 'resnet_4x')
NUM_FEATS = 1175

def diff_pkl_mat():
    with open(os.path.join('test', 'diff_between_pkl_mat2.txt'), 'w') as txt_file:
        for i in range(1, NUM_FEATS+1):
            feat_pkl_fPath = os.path.join(feats_dir, f'fid300_{i:03d}.pkl')
            feat_mat_fPath = os.path.join(feats_dir, f'fid300_{i:03d}.mat')
            
            if i == 1:
                with open(feat_pkl_fPath, 'rb') as feat_file:
                    pkl_dict = pickle.load(feat_file)
                    txt_file.write(f'===== first_feat info in Python =====\n')
                    txt_file.write(f'trace_H, trace_W : {pkl_dict["trace_H"]}, {pkl_dict["trace_W"]}\n')
                    txt_file.write(f'feat_dims: {pkl_dict["feat_dims"]}\n')
                    txt_file.write(f'rfsIm: \n')
                    txt_file.write(f'offset: {pkl_dict["rfsIm"].offset}\n')
                    txt_file.write(f'size: {pkl_dict["rfsIm"].size}\n')
                    txt_file.write(f'stride: {pkl_dict["rfsIm"].stride}\n\n')
                    
                mat_dict = mat73.loadmat(feat_mat_fPath)
                txt_file.write(f'===== first_feat info in MATLAB =====\n')
                txt_file.write(f'trace_H, trace_W : {mat_dict["trace_H"]}, {mat_dict["trace_W"]}\n')
                txt_file.write(f'feat_dims: {mat_dict["feat_dims"]}\n\n')
            
            txt_file.write(f'===== {i} th info =====\n')
            with open(feat_pkl_fPath, 'rb') as feat_file:
                pkl_dict = pickle.load(feat_file)
                txt_file.write(f'In Python version:\n')
                txt_file.write(f'db_feats.shape: {pkl_dict["db_feats"].shape}\n')
                txt_file.write(f'db_labels: {pkl_dict["db_labels"]}\n\n')
            
            mat_dict = mat73.loadmat(feat_mat_fPath)
            txt_file.write(f'In MATLAB version:\n')
            txt_file.write(f'db_feats.shape: {mat_dict["db_feats"].shape}\n')
            txt_file.write(f'db_labels: {mat_dict["db_labels"]}\n\n')