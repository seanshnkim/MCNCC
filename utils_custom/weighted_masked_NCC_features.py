import torch
import numpy as np

# weighted_masked_NCC_features(db_feats, p_ijr_feat, p_ijr_feat_mask, ones_w)
'''
db_feats.shape
(256, 147, 68, 100) = (out_channel_size, height, width, batch_size)

p_ijr_feat.shape
torch.Size([256, 147, 68]) = (out_channel_size, height, width)

p_ijr_feat_mask.shape
(293, 135) = (height, width)

ones_w.shape
torch.Size([1, 1, 68])
'''

def weighted_masked_NCC_features(db, tpl, mask, w_cell):
    N = db.shape[3]
    batchSize = 200
    numBatches = np.ceil(N / batchSize).astype(int)

    all_scores = [np.zeros((1, 1, 1, N), dtype=np.float32) for _ in w_cell]

    for b in range(numBatches):
        inds = slice(b * batchSize, min((b+1) * batchSize, N))
        feats = masked_NCC_features(db[:, :, :, inds], tpl, mask)
        for m, w in enumerate(w_cell):
            scores = np.sum(np.sum(np.convolve(feats, w, mode='valid'), axis=1), axis=2) / feats.shape[2]
            all_scores[m][0, 0, 0, inds] = scores

    return all_scores

def masked_NCC_features(IM, TPL, MASK):
    # MASK is 2D array
    MASK_H, MASK_W = MASK.shape
    TPL_C, TPL_H, TPL_W = TPL.shape
    IM_H, IM_W = IM.shape[1], IM.shape[2]
    
    # assert MASK.shape[0] == TPL.shape[0] and MASK.shape[1] == TPL.shape[1] and MASK.shape[3] == 1
    assert MASK_H == TPL_H and MASK_W == TPL_W
    assert MASK_H == IM_H and MASK_W == IM_W

    nonzero = MASK.sum()
    # IM = IM * MASK  # zero out invalid region
    # IM.shape = (256, 147, 68, 100) and MASK.shape = (147, 68),
    MASK_4D = MASK.unsqueeze(0).unsqueeze(3).expand_as(IM)
    IM = IM * MASK_4D
    
    # mu.shape = (256, 100)
    mu = (IM.mean(axis=1)).mean(axis=1) / nonzero
    mu_4D = mu.unsqueeze(1).unsqueeze(1)
    IM = IM - mu_4D
    IM = IM * MASK_4D  # keep invalid region zero
    # normalized in width and height, respectively
    # Therefore, IM_norm.shape should turn into (256, 100)
    IM_norm = torch.sum(torch.sum(IM**2, axis=1), axis=1)

    # TPL.shape = (256, 147, 68) MASK.shape = (147, 68)
    TPL = TPL * MASK
    # mu = np.sum(np.sum(TPL, axis=1), axis=2) / nonzero
    mu = (TPL.sum(axis=1)).sum(axis=1) / nonzero
    mu_3D = mu.unsqueeze(1).unsqueeze(2)
    TPL = TPL - mu_3D
    TPL = TPL * MASK
    TPL_norm = torch.sum(torch.sum(TPL**2, axis=1), axis=1)

    TPL_4D = TPL.unsqueeze(3)
    numer = IM * TPL_4D
    denom = IM_norm * TPL_norm.unsqueeze(1) + 1e-5
    feat = numer / denom.unsqueeze(1).unsqueeze(2)
    # MASK.shape = torch.Size([147, 68])
    feat = feat * MASK.unsqueeze(0).unsqueeze(3)

    return feat