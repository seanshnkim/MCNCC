import numpy as np
import pickle
import os
import scipy.io as sio

from utils_custom import get_db_attrs


def basel_compare_cmc_latent_masked(db_ind=2):
    NUM_IMAGES = 1175
    """Compares the CMC of different methods on the FID-300 dataset."""

    _, db_chunk_inds, dbname = get_db_attrs('fid300', db_ind, {'suffix'})
    start_idx, end_idx = db_chunk_inds
    
    db_labels = np.zeros(end_idx-start_idx, dtype=np.int32)
    for idx in range(start_idx, end_idx):
        feat_path = os.path.join('feats', dbname, f'fid300_{idx:03d}.pkl')
        db_labels[idx-1] = pickle.load(open(feat_path, 'rb'))['db_labels'][0, 0, 0, 0]
    
    label_path = os.path.join('datasets', 'FID-300', 'label_table.mat')
    label_table = sio.loadmat(label_path)['label_table']
    
    ncc_cmc = np.zeros(NUM_IMAGES, dtype=np.float32)
    for p in range(300):
        # Load patch results.
        
        # Python (pickle file) version
        # fname = os.path.join('results', dbname, f'fid300_alignment_search_ones_res_{p:04d}.pkl')
        # with open(fname, 'rb') as f:
        #     minsONES = pickle.load(f)
        fname = os.path.join('results', dbname, f'fid300_alignment_search_ones_res_{p:04d}.mat')
        minsONES = sio.loadmat(fname)['minsONES']

        query_label = label_table[p, 1]

        # NCC.
        inds = np.argsort(minsONES.flatten(), kind='stable')[::-1]
        ncc_cmc += np.cumsum(inds == query_label)

    ncc_cmc = ncc_cmc / 300 * 100

    baselines = ['datasets/FID-300/result_ACCV14.mat', 'datasets/FID-300/ranks_BMVC16.mat', 'datasets/FID-300/ranks_LoG16.mat']
    base_cmc = np.zeros((len(baselines), 1175), dtype=np.float32)
    for b in range(len(baselines)):
        with open(baselines[b], 'rb') as f:
        ranks = pickle.load(f)['ranks']

        for p in range(300):
        res = np.zeros(1175, dtype=np.bool_)
        res[ranks[p]] = True
        base_cmc[b] += np.cumsum(res)

        base_cmc[b] = base_cmc[b] / 300 * 100

    return ncc_cmc, base_cmc


if __name__ == '__main__':
  ncc_cmc, base_cmc = basel_compare_cmc_latent_masked()

  # Plot the CMC curves.
  import matplotlib.pyplot as plt

  plt.plot(ncc_cmc, label='NCC')
  for b in range(len(baselines)):
    plt.plot(base_cmc[b], label=baselines[b])

  plt.xlabel('Rank')
  plt.ylabel('Match rate (%)')
  plt.legend()
  plt.title('CMC comparison on the FID-300 dataset')
  plt.show()