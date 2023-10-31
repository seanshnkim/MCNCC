import numpy as np
import cv2


# feat_dims = (1175, 256, 147, 68) = (batch_size, out_channel_size, height, width)
def warp_masks(print_masks, im_f2i, feat_dims, db_ind):
    feat_H, feat_W = feat_dims[2], feat_dims[3]
    
    if db_ind == 0:
        feat_masks = print_masks
    else:
        # Inverting the transformation matrix
        im_f2i_inv = np.linalg.inv(im_f2i)
        
        #REVIEW Shape of feat_masks should be converted into (feat_H, feat_W) after warping (in MATLAB code)
        feat_masks = cv2.warpPerspective(print_masks, im_f2i_inv, \
            (feat_W, feat_H), flags=cv2.INTER_NEAREST)
    
    return feat_masks
