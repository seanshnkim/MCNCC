def get_db_attrs(dataset, db_ind, info=None):
    if info is None:
        info_inds = [0, 1, 2]
    else:
        all_info = ['suffix', 'layer', 'model']
        info_inds = [i for i, val in enumerate(all_info) if val in info]
    
    # db_attr = {suffix, layer to cut network, pool size, pool stride, pool pad, img pad}
    db_attrs_options = [
        ('pixel_raw', 'conv1', ''),
        ('resnet_2x', 'pool1', 'imagenet-resnet-50-dag.mat'),
        ('resnet_4x', 'res2c_branch2a', 'imagenet-resnet-50-dag.mat'),
        ('resnet_8x', 'res3d_branch2a', 'imagenet-resnet-50-dag.mat'),
        ('resnet_16x', 'res4d_branch2a', 'imagenet-resnet-50-dag.mat'),
        ('googlenet_4x', 'norm2', 'imagenet-googlenet-dag.mat'),
        ('vgg_4x', 'conv2', 'imagenet-vgg-m.mat')
    ]
    
    db_attr = db_attrs_options[db_ind]
    dbname = db_attr[0]
    
    if dataset.lower() == 'israeli':
        db_chunks = [list(range(1, 388))]
    elif dataset.lower() == 'fid300':
        db_chunks = [list(range(1, 1176))]
    elif dataset.lower() == 'facades':
        db_chunks = [list(range(1, 1658))]
    elif dataset.lower() == 'maps':
        db_chunks = [list(range(1, 1098)), list(range(1098, 2195))]
    else:
        raise ValueError(f'Dataset: {dataset} is not valid!')
    
    db_attr = [db_attr[i] for i in info_inds]
    
    return db_attr, db_chunks, dbname

# You can call the function like this:
# attrs, chunks, name = get_db_attrs('fid300', 2)
