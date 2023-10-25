import numpy as np
from skimage.transform import estimate_transform
from getVarReceptiveFields_custom import RecetiveField

def feat_2_image(rfs):
    if isinstance(rfs, RecetiveField):
        rfs = [rfs]

    feat_coords_x1 = [-100, -100, +100, +100]
    feat_coords_y1 = [-100, +100, -100, +100]    
    feat_coords_x2 = [-100, -100, +100, +100]
    feat_coords_y2 = [-100, +100, -100, +100]
    
    # In MATLAB code, rfs was 1*n struct array but here it is just a single instance
    for rf in rfs:
        if isinstance(rf, list) and len(rf) > 1:
            rf = rf[-1]

        feat_coords_x2 = rf.stride[1] * (np.array(feat_coords_x2) - 1) + rf.offset[1]
        feat_coords_y2 = rf.stride[0] * (np.array(feat_coords_y2) - 1) + rf.offset[0]
    
    # feat_coords_x2 = rfs.stride[1] * (np.array(feat_coords_x2) - 1) + rfs.offset[1]
    # feat_coords_y2 = rfs.stride[0] * (np.array(feat_coords_y2) - 1) + rfs.offset[0]

    src = np.column_stack((feat_coords_x1, feat_coords_y1))
    dst = np.column_stack((feat_coords_x2, feat_coords_y2))
    
    tform = estimate_transform('affine', src, dst)
    
    return tform.params
