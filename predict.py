import numpy as np
import pickle
import os
import pandas as pd
import scipy.io as sio

from utils_custom.get_db_attrs import get_db_attrs

NUM_QUERIES = 300

def predict_top10(db_ind=2):
    """Compares the CMC of different methods on the FID-300 dataset."""
    _, db_chunk_inds, dbname = get_db_attrs('fid300', db_ind, {'suffix'})
    start_idx, end_idx = db_chunk_inds[0]
    
    NUM_REF = end_idx - start_idx
    
    label_path = os.path.join('datasets', 'FID-300', 'label_table.csv')
    label_table = pd.read_csv(label_path, header=None)
    
    new_dbname = 'resnet_4x_no_align'
    # ncc_cmc = np.zeros(NUM_REF, dtype=np.float32)
    
    predicted = np.zeros((NUM_QUERIES, ), dtype=np.bool_)
    for qidx in range(NUM_QUERIES):
        score_save_fname = os.path.join('results', new_dbname, f'fid300_ones_res_{qidx+1:04d}.npz')
        with np.load(score_save_fname) as scores:
            # minsONES.shape = (1175, 1, 1, 1)
            minsONES = scores['scores']

        query_label = label_table.iloc[qidx, 1]

        # NCC.
        pred_inds = np.argsort(minsONES.flatten(), kind='stable')
        # indices starting from 1 (because label_table is compatible with MATLAB code)
        pred_inds += 1
        
        pred_top10 = pred_inds[-10:]
        if query_label in pred_top10:
            predicted[qidx] = True

    return predicted