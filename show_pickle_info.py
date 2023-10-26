import pickle
import os

feats_dir = os.path.join('feats', 'resnet_4x')

# Write in a text fil
with open('pickle_info.txt', 'w') as txt_file:
    for i in range(1, 1176):
        feat_fPath = os.path.join(feats_dir, f'fid300_{i:03d}.pkl')
        with open(feat_fPath, 'rb') as feat_file:
            dat = pickle.load(feat_file)
            txt_file.write(f'===== {i} th feat =====\n')
            #  write dat['trace_H'] in the text file
            txt_file.write(f'trace_H: {dat["trace_H"]}\n')
            #  write dat['trace_W'] in the text file
            txt_file.write(f'trace_W: {dat["trace_W"]}\n')
            txt_file.write('\n')