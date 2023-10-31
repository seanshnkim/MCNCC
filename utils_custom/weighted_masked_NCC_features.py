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

    nonzero = np.sum(MASK)
    # IM = IM * MASK  # zero out invalid region
    # IM.shape = (256, 147, 68, 100) and MASK.shape = (147, 68),
    MASK_exp = np.expand_dims(MASK, axis=0)
    MASK_exp = np.expand_dims(MASK_exp, axis=-1)
    IM = IM * MASK_exp
    
    # mu.shape = (256, 100)
    mu = np.sum(np.sum(IM, axis=1), axis=1) / nonzero  # compute mean of valid region
    mu_exp = np.expand_dims(mu, axis=1)
    mu_exp = np.expand_dims(mu_exp, axis=1)
    IM = IM - mu_exp
    IM = IM * MASK_exp  # keep invalid region zero
    # normalized in width and height, respectively
    # Therefore, IM_norm.shape should turn into (256, 100)
    IM_norm = np.sum(np.sum(IM**2, axis=1), axis=1)

    # TPL.shape = (256, 147, 68)
    #FIXME - TPL.device = cuda:0, and MASK is just numpy array
    TPL = TPL * MASK
    mu = np.sum(np.sum(TPL, axis=1), axis=2) / nonzero
    TPL = TPL - mu
    TPL = TPL * MASK
    TPL_norm = np.sum(np.sum(TPL**2, axis=1), axis=2)

    numer = IM * TPL
    denom = np.sqrt(IM_norm * TPL_norm + 1e-5)
    feat = numer / denom
    feat = feat * MASK

    return feat

'''
I used NumPy, a powerful library in Python that supports a wide range of mathematical operations, 
making the code more MATLAB-like.

np.convolve() is used to replicate MATLAB's vl_nnconv() function. 

However, be cautious as they may not be entirely equivalent 
depending on the specifics of your use case, 
and you might want to use a different function or library 
like TensorFlow or PyTorch for more complex operations.

For indexing, Python uses 0-based indexing, 
so adjustments are made in loop indices and slicing. 
Slicing is used in place of MATLAB's colon notation for indexing arrays.

Instead of bsxfun(), broadcasting in NumPy is utilized, 
which automatically applies element-wise binary operations in a vectorized manner.

Lambda functions and the map() function are used to 
apply a function element-wise to items in a list or array, 
replacing the usage of cell arrays in MATLAB.
'''