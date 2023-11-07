function rotate_pad_queries(p_inds, db_ind)
% seems faster to run database against latent prints, potentially fewer
% translations to look over (especially when latent prints are MUCH smaller than
% test impressions)
imscale = 0.5;


% To use 'utils/feat_2_image.m'
addpath('./utils_custom');

[~, ~, dbname] = get_db_attrs('fid300', db_ind);
load(fullfile('feats', dbname, 'fid300_001.mat'), ...
  'trace_H', 'trace_W')

load(fullfile('results', 'latent_ims_mean_pix.mat'), 'mean_im_pix')

for p=reshape(p_inds, 1, [])
  p_im = imresize(imread(fullfile('datasets', 'FID-300', 'tracks_cropped', sprintf('%05d.jpg', p))), ...
                  imscale);
  [p_H, p_W] = size(p_im);
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
  [p_H, p_W, ~] = size(p_im);

  % pad latent print so that for every pixel location (of the original latent print)
  % we can extract an image of the same size as the test impressions
  pad_H = trace_H-p_H; pad_W = trace_W-p_W;
  p_im_padded = padarray(p_im, [pad_H pad_W], 255, 'both');
  p_mask_padded = padarray(ones(p_H, p_W, 'logical'), [pad_H pad_W], 0, 'both');

  angles = -20:4:20;
  p_save_dir = fullfile('results', 'resnet_4x_matlab', sprintf('%04d', p));
  mkdir(p_save_dir);

  for r=1:numel(angles)
    % rotate image and mask
    p_im_padded_r = imrotate(p_im_padded, angles(r), 'bicubic', 'crop');
    p_mask_padded_r = imrotate(p_mask_padded, angles(r), 'nearest', 'crop');
    
    imwrite(p_im_padded_r, fullfile(p_save_dir, sprintf('fid300_rotated_im_%04d_%03d.jpg', p, angles(r))));
    imwrite(p_mask_padded_r, fullfile(p_save_dir, sprintf('fid300_rotated_mask_%04d_%03d.jpg', p, angles(r))));

  end
end
