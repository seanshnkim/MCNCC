ims = rand(3,4,5,6);
ims
gray = zeros(size(ims, 1), size(ims, 2), 3, size(ims, 4), 'like', ims);
for i=1:size(ims, 4)
  gray(:,:,:, i) = repmat(mean(ims(:,:,:, i), 3), 1, 1, 3);
end