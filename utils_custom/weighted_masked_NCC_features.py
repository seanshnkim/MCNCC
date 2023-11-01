import torch
import torch.nn.functional as F
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

# w_cell.shape = (1, 1, 256)
def weighted_masked_NCC_features(db, tpl, mask, w_cell):
    # entire data_size N
    N = db.shape[3]
    batchSize = 200
    numBatches = np.ceil(N / batchSize).astype(int)

    all_scores = torch.zeros((1, 1, 1, N), dtype=torch.float32)

    for b in range(numBatches):
        inds = slice(b * batchSize, min((b+1) * batchSize, N))
        feats = masked_NCC_features(db[:, :, :, inds], tpl, mask)
        
        # torch.Size([out_channels=1, in_channels=256, kernel_height=1, kernel_width=1])
        w_cell_expanded = w_cell.unsqueeze(0)
        # feats_permuted.shape = torch.Size([minibatch=100, in_channels=256, height=147, width=68])
        feats_permuted = feats.permute(3, 0, 1, 2)
        # conv_result.shape = torch.Size([100, 1, 147, 68])
        conv_result = F.conv2d(feats_permuted, w_cell_expanded, padding='valid')
        # Summing over the width and height dimensions and normalizing by the minibatch size
        all_scores[0, 0, 0, inds] = conv_result.sum(dim=(2, 3), keepdim=True).div(feats.size(3)).permute(1, 2, 3, 0)

    return all_scores


def masked_NCC_features(IM, TPL, MASK):
    # MASK is 2D array
    MASK_H, MASK_W = MASK.shape
    TPL_C, TPL_H, TPL_W = TPL.shape
    IM_H, IM_W = IM.shape[1], IM.shape[2]
    
    assert MASK_H == TPL_H and MASK_W == TPL_W
    assert MASK_H == IM_H and MASK_W == IM_W

    nonzero = MASK.sum()
    # IM.shape = (256, 147, 68, 100) and MASK.shape = (147, 68)
    MASK_4D = MASK.unsqueeze(0).unsqueeze(3).expand_as(IM)
    # IM = IM * MASK_4D -> Instead, use in-place multiplication to optimize GPU memory usage
    IM.mul_(MASK_4D)
    
    # mu.shape = (256, 100)
    mu = IM.mean(dim=(1,2)) / nonzero
    mu_4D = mu.unsqueeze(1).unsqueeze(1)
    IM.sub_(mu_4D).mul_(MASK_4D)
    # normalized in width and height, respectively
    # Therefore, IM_norm.shape should turn into (256, 100)
    IM_norm = IM.pow(2).sum(dim=(1,2))

    # TPL.shape = (256, 147, 68) MASK.shape = (147, 68) -> broadcasted, TPL.shape = (256, 147, 68)
    TPL.mul_(MASK)
    mu_TPL = TPL.sum(dim=(1,2)) / nonzero
    mu_TPL_3D = mu_TPL.unsqueeze(1).unsqueeze(2)
    TPL.sub_(mu_TPL_3D).mul_(MASK)
    # TPL_norm = torch.sum(torch.sum(TPL**2, axis=1), axis=1)
    TPL_norm = TPL.pow(2).sum(dim=(1,2))

    TPL_4D = TPL.unsqueeze(3)
    numer = IM * TPL_4D
    denom = (IM_norm * TPL_norm.unsqueeze(1) + 1e-5).unsqueeze(1).unsqueeze(2)
    feat = numer / denom
    feat.mul_(MASK.unsqueeze(0).unsqueeze(3))
    
    return feat