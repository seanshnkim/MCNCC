import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

# assume the networks last variable (x) is the features
# if x is HxWxKxN, then there are H*W number of patches for each data point
# identical to generate_db_CNNfeats, but leaves results on the GPU

def generate_db_CNNfeats_gpu(model, img, batch_size=10, device='cuda'):
    # Set the model to evaluation mode
    model.eval()
    
    # Move the model to the GPU
    model.to(device)
    
    # Get the data and convert it to grayscale
    # ims = gather(data{2});
    # ims = data.cpu().numpy()
    
    gray = np.mean(img, axis=2, keepdims=True).repeat(3, axis=2)
    # change into 4D array for batch_size dimension
    gray_expanded = np.expand_dims(gray, axis=3)
    gray_expanded = np.transpose(gray_expanded, (3, 2, 0, 1))
    
    # Convert grayscale images to torch tensor and move it to the GPU
    img_gpu = torch.tensor(gray_expanded, dtype=torch.float32).to(device)
    
    # Calculate the number of batches
    data_size = gray_expanded.shape[0]
    num_batches = int(np.ceil(data_size / batch_size))
    
    feats = []
    # Process each batch
    for b in range(num_batches):
        left = b * batch_size
        right = min((b + 1) * batch_size, data_size)
        
        # Forward pass
        with torch.no_grad():
            # Given groups=1, weight of size [64, 3, 7, 7], 
            # expected input[433, 168, 3, 1] to have 3 channels, but got 168 channels instead
            outputs = model(img_gpu[left:right, :, :, :])
        
        # Collecting the features (responses)
        feats.append(outputs)
    
    # Concatenating all the features
    # can't convert cuda:0 device type tensor to numpy. 
    # Use Tensor.cpu() to copy the tensor to host memory first.
    for i in range(len(feats)):
        feats[i] = feats[i].cpu().numpy()
    db = np.concatenate(feats, axis=0)
    db_gpu = torch.tensor(db, dtype=torch.float32).to(device)
    
    return db_gpu
