function alignment_search_eval_fid300_test(p_inds, db_ind)
% seems faster to run database against latent prints, potentially fewer
% translations to look over (especially when latent prints are MUCH smaller than
% test impressions)
imscale = 0.5;
erode_pct = 0.1;

if nargin<2
  db_ind = 2;
end

% To use 'utils/feat_2_image.m'
addpath('./utils_custom');

[db_attr, db_chunks, dbname] = get_db_attrs('fid300', db_ind);


% load and modify network
net = dagnn.DagNN();
if db_ind==0
  net.addLayer('identity', dagnn.Conv('size', [1 1 3 1], ...
                                      'stride', 1, ...
                                      'pad', 0, ...
                                      'hasBias', false), ...
               {'data'}, {'raw'}, {'I'});
  net.params(1).value = reshape(single([1 0 0]), 1, 1, 3, 1);
else
  flatnn = load(fullfile('models', db_attr{3}));
  net = net.loadobj(flatnn);
  ind = net.getLayerIndex(db_attr{2});
  net.layers(ind:end) = []; net.rebuild();
end
load(fullfile('results', 'latent_ims_mean_pix.mat'), 'mean_im_pix')
mean_im_pix = mean_im_pix; % stupid MATLAB transparency



% load database chunk
db_chunk_inds = db_chunks{1};
load(fullfile('feats', dbname, 'fid300_001.mat'), ...
  'db_feats', 'feat_dims', 'rfsIm', 'trace_H', 'trace_W')
db_feats = zeros(size(db_feats, 1), size(db_feats, 2), size(db_feats, 3), ...
  numel(db_chunk_inds), 'like', db_feats);
for i=1:numel(db_chunk_inds)
  dat = load(fullfile('feats', dbname, sprintf('fid300_%03d.mat', db_chunk_inds(i))));
  db_feats(:,:,:, i) = dat.db_feats;
end

% I added this (not original part)
% rfs = net.getVarReceptiveFields(1);
% rfsIm = rfs(end);

im_f2i = feat_2_image(rfsIm);

radius = max(1, floor(min(feat_dims(1), feat_dims(2))*erode_pct));
se = strel('disk', radius, 0);

ones_w = gpuArray.ones(1, 1, feat_dims(3), 'single');

db_feats = gpuArray(db_feats);
for p=reshape(p_inds, 1, [])
  % fname = fullfile('results', dbname, sprintf('fid300_alignment_search_ones_res_%04d.mat', p));
  % if exist(fname, 'file'), continue, end
  % lock_fname = [fname, '.lock'];
  % if exist(lock_fname, 'file'), continue, end
  % fid = fopen(lock_fname, 'w');
  % fprintf('p=%d: ', p ),tic

  p_im = imresize(imread(fullfile('datasets', 'FID-300', 'tracks_cropped', sprintf('%05d.jpg', p))), ...
                  imscale);
  [p_H, p_W] = size(p_im);
  % fix latent prints are bigger than the test impressions
  if p_H>p_W
    if p_H>trace_H
      p_im = imresize(p_im, [trace_H NaN]);
    end
  else
    if p_W>trace_W
      p_im = imresize(p_im, [NaN trace_W]);
    end
  end
  
  % Subtract mean_im_pix from p_im
  p_im = bsxfun(@minus, single(p_im), mean_im_pix);
  [p_H, p_W, p_C] = size(p_im);

  % pad latent print so that for every pixel location (of the original latent print)
  % we can extract an image of the same size as the test impressions
  pad_H = trace_H-p_H; pad_W = trace_W-p_W;
  p_im_padded = padarray(p_im, [pad_H pad_W], 255, 'both');
  p_mask_padded = padarray(ones(p_H, p_W, 'logical'), [pad_H pad_W], 0, 'both');

  angles = -20:4:20;

  for r=1:numel(angles)
    % rotate image and mask
    p_im_padded_r = imrotate(p_im_padded, angles(r), 'bicubic', 'crop');
    p_mask_padded_r = imrotate(p_mask_padded, angles(r), 'nearest', 'crop');
    
    imwrite(p_im_padded_r, fullfile('results', 'resnet_4x_matlab', sprintf('fid300_rotated_im_%04d_%03d.jpg', p, angles(r))));
    imwrite(p_mask_padded_r, fullfile('results', 'resnet_4x_matlab', sprintf('fid300_rotated_mask_%04d_%03d.jpg', p, angles(r))));

  end
end
