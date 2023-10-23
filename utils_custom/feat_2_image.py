import numpy as np
from skimage.transform import estimate_transform

class RFStruct:
    def __init__(self, size, stride, offset):
        self.size = size
        self.stride = stride
        self.offset = offset

def feat_2_image(rfs):
    if isinstance(rfs, RFStruct):
        rfs = [rfs]

    feat_coords_x1 = [-100, -100, +100, +100]
    feat_coords_y1 = [-100, +100, -100, +100]    
    feat_coords_x2 = [-100, -100, +100, +100]
    feat_coords_y2 = [-100, +100, -100, +100]
    
    for rf in rfs:
        if isinstance(rf, list) and len(rf) > 1:
            rf = rf[-1]

        feat_coords_x2 = rf.stride[1] * (np.array(feat_coords_x2) - 1) + rf.offset[1]
        feat_coords_y2 = rf.stride[0] * (np.array(feat_coords_y2) - 1) + rf.offset[0]

    src = np.column_stack((feat_coords_x1, feat_coords_y1))
    dst = np.column_stack((feat_coords_x2, feat_coords_y2))
    
    tform = estimate_transform('affine', src, dst)
    
    return tform.params

# Example usage:
rfs_example = [RFStruct(size=[11, 11], stride=[4, 4], offset=[3, 3])] * 27  # 1x27 struct array
transformation_matrix = feat_2_image(rfs_example)
print(transformation_matrix)
