from gen_feats_fid300 import gen_feats_fid300
from alignment_search_eval_fid300 import alignment_search_eval_fid300

# gen_feats_fid300(2)
#NOTE - MATLAB follows math conventions, so the first index is 1, not 0
# However, in get_db_arrs.m first case starts with 0
# gen_feats_fid300(2)

alignment_search_eval_fid300([1, 300], db_ind=2)