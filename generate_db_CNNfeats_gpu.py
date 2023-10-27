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
    
    # Convert grayscale images to torch tensor and move it to the GPU
    img_cpu = torch.tensor(gray_expanded, dtype=torch.float32).to(device)
    
    # Calculate the number of batches
    num_batches = int(np.ceil(gray_expanded.shape[3] / batch_size))
    
    feats = []
    # Process each batch
    for b in range(num_batches):
        left = b * batch_size
        right = min((b + 1) * batch_size, gray_expanded.shape[3])
        
        # Forward pass
        with torch.no_grad():
            outputs = model(img_cpu[:, :, :, left:right])
        
        # Collecting the features (responses)
        feats.append(outputs)
    
    # Concatenating all the features
    db = np.concatenate(feats, axis=0)
    
    return db
