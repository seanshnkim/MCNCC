import cv2
import numpy as np

def warp_masks(print_masks, im_f2i, feat_dims, db_ind):
    if db_ind == 0:
        feat_masks = print_masks
    else:
        # Applying the inverse warp transformation
        # Assuming im_f2i.invert() returns an inverted transformation matrix
        inv_transform = cv2.invertAffineTransform(im_f2i.invert()) 
        feat_masks = cv2.warpAffine(print_masks, inv_transform, (print_masks.shape[1], print_masks.shape[0]), flags=cv2.INTER_NEAREST)
        
        # Adjusting the size of feat_masks to match feat_dims if necessary
        if feat_masks.shape[0] != feat_dims[0] or feat_masks.shape[1] != feat_dims[1]:
            
            if feat_masks.shape[0] > feat_dims[0]:
                top = 1  # Adjust index for Python's 0-based indexing
                bot = feat_dims[0]
            else:
                top = 0
                bot = feat_dims[0] - 1  # Adjust to ensure it gets the correct ending index
            
            if feat_masks.shape[1] > feat_dims[1]:
                lef = 1  # Adjust index for Python's 0-based indexing
                rig = feat_dims[1]
            else:
                lef = 0
                rig = feat_dims[1] - 1  # Adjust to ensure it gets the correct ending index
            
            feat_masks = feat_masks[top:bot, lef:rig, ...]  # "..." is used to include any remaining dimensions
            
    return feat_masks