import numpy as np

def verify_score_results(query_ind):
    for i in range(query_ind[0], query_ind[1]+1):
        scores = np.load(f"results/resnet_4x_no_align/fid300_ones_res_{i:04d}.npz")
        for fname in scores.files:
            if np.isnan(scores[fname]).any() or np.isinf(scores[fname]).any():
                return False
    return True