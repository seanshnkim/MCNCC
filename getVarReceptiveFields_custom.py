import numpy as np
import torch
import torch.nn as nn
from get_num_conv import get_num_conv

class RecetiveField:
    def __init__(self, size=np.array([]), stride=np.array([]), offset=np.array([])):
        self.size = size
        self.stride = stride
        self.offset = offset


def get_receptive_fields(obj):
    # net = ModifiedNetwork(db_ind=2, db_attr=db_attr)
    # net.model[0].kernel_size  # (7, 7)
    ks = obj.kernel_size
    # net.model[0].dilation # (1, 1) -> return dilation ratio
    # if dilation ratio = (1, 1), it means no dilation (default value)
    ke = (np.array(ks) - 1) * np.array(obj.dilation) + 1
    
    y1 = 1 - obj.padding[0]
    y2 = 1 - obj.padding[0] + ke[0] - 1
    x1 = 1 - obj.padding[2]
    x2 = 1 - obj.padding[2] + ke[1] - 1
    
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    
    return RecetiveField(size=[h, w], stride=obj.stride, offset=[(y1+y2)/2, (x1+x2)/2])


def compose_receptive_fields(rf1, rf2):
    if rf1.size.size == 0 or rf2.size.size == 0:
        return RecetiveField()

    rf = RecetiveField()
    rf.size = rf1.stride * (rf2.size - 1) + rf1.size
    rf.stride = rf1.stride * rf2.stride
    rf.offset = rf1.stride * (rf2.offset - 1) + rf1.offset
    return rf


def resolve_receptive_fields(rfs):
    rf = RecetiveField()

    for rf_i in rfs:
        if rf_i.size.size == 0:
            continue
        if np.isnan(rf_i.size).any():
            rf.size = np.array([np.nan, np.nan])
            rf.stride = np.array([np.nan, np.nan])
            rf.offset = np.array([np.nan, np.nan])
            break
        if rf.size.size == 0:
            rf = rf_i
        else:
            if not np.array_equal(rf.stride, rf_i.stride):
                rf.size = np.array([np.nan, np.nan])
                rf.stride = np.array([np.nan, np.nan])
                rf.offset = np.array([np.nan, np.nan])
                break
            else:
                a = rf.offset - (rf.size - 1) / 2
                b = rf.offset + (rf.size - 1) / 2
                c = rf_i.offset - (rf_i.size - 1) / 2
                d = rf_i.offset + (rf_i.size - 1) / 2
                e = np.minimum(a, c)
                f = np.maximum(b, d)
                rf.offset = (e + f) / 2
                rf.size = f - e + 1
    return rf


# def getVarReceptiveField(obj):
#     num_convLayer = get_num_conv(obj)
#     for 