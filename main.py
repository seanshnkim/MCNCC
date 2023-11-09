# from gen_feats_fid300 import gen_feats_fid300, preprocess_im
# from alignment_search_eval_fid300 import alignment_search_eval_fid300
# from diff_between_pkl_mat import diff_pkl_mat
from predict import predict_top10

# gen_feats_fid300(2)
# preprocess_im(0,0)
# alignment_search_eval_fid300([9, 11], db_ind=2)
# diff_pkl_mat()

# from eval_fid300 import eval_fid300
# eval_fid300([9, 11], 2)

predicted = predict_top10(2)

print("This is a test code run by resnet50 backbone. No model training included.")
print(f"Top 10 Accuracy: {(predicted.sum() / predicted.shape[0]):.4f}")