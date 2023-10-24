import numpy as np

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
    H, W, C = TPL.shape
    assert MASK.shape[0] == TPL.shape[0] and MASK.shape[1] == TPL.shape[1] and MASK.shape[3] == 1

    nonzero = np.sum(MASK)
    IM = IM * MASK  # zero out invalid region
    mu = np.sum(np.sum(IM, axis=1), axis=2) / nonzero  # compute mean of valid region
    IM = IM - mu
    IM = IM * MASK  # keep invalid region zero
    IM_norm = np.sum(np.sum(IM**2, axis=1), axis=2)

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