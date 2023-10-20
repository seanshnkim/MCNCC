import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

# assume the networks last variable (x) is the features
# if x is HxWxKxN, then there are H*W number of patches for each data point
# identical to generate_db_CNNfeats, but leaves results on the GPU

def generate_db_CNNfeats_gpu(model, data, batch_size=10, device='cuda'):
    # Set the model to evaluation mode
    model.eval()
    
    # Move the model to the GPU
    model.to(device)
    
    # Get the data and convert it to grayscale
    # ims = gather(data{2});
    ims = data[1].cpu().numpy()
    
    gray = np.mean(ims, axis=2, keepdims=True).repeat(3, axis=2)
    
    # Convert grayscale images to torch tensor and move it to the GPU
    data[1] = torch.tensor(gray, dtype=torch.float32).to(device)
    
    # Calculate the number of batches
    num_batches = int(np.ceil(len(gray) / batch_size))
    
    feats = []
    # Process each batch
    for b in range(num_batches):
        left = b * batch_size
        right = min((b + 1) * batch_size, len(gray))
        
        # Forward pass
        with torch.no_grad():
            outputs = model(data[1][left:right])
        
        # Collecting the features (responses)
        feats.append(outputs)
    
    # Concatenating all the features
    db = np.concatenate(feats, axis=0)
    
    return db

# Usage example:
# model - your PyTorch model
# data - your input data as a list where data[1] contains the image tensors
# db = generate_db_CNNfeats(model, data)
