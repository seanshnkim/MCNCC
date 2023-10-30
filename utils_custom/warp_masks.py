import numpy as np
import cv2

def warp_masks(print_masks, im_f2i, feat_dims, db_ind):
    if db_ind == 0:
        feat_masks = print_masks
    else:
        # Inverting the transformation matrix
        im_f2i_inv = np.linalg.inv(im_f2i)
        
        # Applying the warp
        rows, cols = print_masks.shape
        feat_masks = cv2.warpPerspective(print_masks, im_f2i_inv, (cols, rows), flags=cv2.INTER_NEAREST)
        
        # Cropping or padding feat_masks if it does not match feat_dims
        if feat_masks.shape[0] != feat_dims[0] or feat_masks.shape[1] != feat_dims[1]:
            if feat_masks.shape[0] > feat_dims[0]:
                top = 1
                bot = feat_dims[0]
            else:
                top = 0
                bot = feat_dims[0] - 1
            
            if feat_masks.shape[1] > feat_dims[1]:
                left = 1
                right = feat_dims[1]
            else:
                left = 0
                right = feat_dims[1] - 1
                
            feat_masks = feat_masks[top:bot, left:right]
    
    return feat_masks

# Test the function
print_masks = np.random.rand(293, 135)
im_f2i = np.random.rand(3, 3)
feat_dims = (1175, 256, 147, 68)
db_ind = 2

feat_masks = warp_masks(print_masks, im_f2i, feat_dims, db_ind)
