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
def weighted_masked_NCC_features(db_feats, query_feat, mask, weight):
    # entire data_size N
    db_chunk_size = db_feats.shape[0]
    BATCH_SIZE = 20
    num_batches = np.ceil(db_chunk_size / BATCH_SIZE).astype(int)

    all_scores = torch.zeros((db_chunk_size, 1, 1, 1), dtype=torch.float32)

    for b in range(num_batches):
        inds = slice(b * BATCH_SIZE, min((b+1) * BATCH_SIZE, db_chunk_size))
        feats = masked_NCC_features(db_feats[inds, :, :, :], query_feat, mask)
        # weight.shape = torch.Size([1, channels=256, ksize=1, ksize=1])
        # conv_result.shape = torch.Size([BATCH_SIZE, 1, height, width])
        conv_result = F.conv2d(feats, weight, padding='valid')
        # Sum over the width and height dimensions and normalize by the minibatch size
        # length of inds(slice object) and feats.shape[0] should be the same
        all_scores[inds, :, :, :] = conv_result.sum(dim=(2, 3), keepdim=True).div(feats.shape[0])

    return all_scores


def masked_NCC_features(imgs, feat, mask):
    # MASK is 2D array 
    mask_H, mask_W = mask.shape
    feat_CH, feat_H, feat_W = feat.shape
    img_H, img_W = imgs.shape[2], imgs.shape[3]
    
    assert mask_H == feat_H and mask_W == feat_W
    assert mask_H == img_H and mask_W == img_W

    nonzero = mask.sum()
    # img.shape = (batch_size=100, channels=256, height=147, width=68) and mask.shape = (height=147, width=68)
    mask_4D = mask.expand_as(imgs)
    # IM = IM * MASK_4D -> Instead, use in-place multiplication to optimize GPU memory usage
    imgs.mul_(mask_4D)
    
    # get mean of each image by summing over width and height and divide by "nonzero"
    mu = torch.mean(imgs, dim=(2,3), keepdim=True) / nonzero
    imgs.sub_(mu).mul_(mask_4D)
    # normalize in width and height, respectively. IM_norm shape should be (256, 100)
    imgs_norm = imgs.pow(2).sum(dim=(2, 3))

    # feat.shape = (channels=256, height=147, width=68) mask.shape = (147, 68)
    feat.mul_(mask)
    # get mean of each feature by summing over width and height and divide by "nonzero"
    mu_feat = feat.sum(dim=(1, 2)) / nonzero
    mu_feat = mu_feat.unsqueeze(-1).unsqueeze(-1)
    feat.sub_(mu_feat).mul_(mask)
    feat_norm = feat.pow(2).sum(dim=(1,2))

    numer = imgs * feat
    denom = torch.sqrt(imgs_norm * feat_norm.unsqueeze(0) + 1e-5)
    feat = numer / denom.unsqueeze(-1).unsqueeze(-1)
    feat.mul_(mask)
    
    return feat