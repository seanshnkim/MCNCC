from gen_feats_fid300 import gen_feats_fid300
# from alignment_search_eval_fid300_test import alignment_search_eval_fid300_test
from alignment_search_eval_fid300 import alignment_search_eval_fid300
# from alignment_search_eval_fid300 import alignment_search_eval_fid300
import sys
# from diff_between_pkl_mat import diff_pkl_mat

# gen_feats_fid300(2)
#NOTE - MATLAB follows math conventions, so the first index is 1, not 0
# However, in get_db_arrs.m first case starts with 0

# sys.argv
# alignment_search_eval_fid300([sys.argv[1], sys.argv[2]], db_ind=2)
# alignment_search_eval_fid300_test([1, 3], db_ind=2)
alignment_search_eval_fid300([1, 3], db_ind=2)

# diff_pkl_mat()