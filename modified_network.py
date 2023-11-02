import torch
import torch.nn as nn
from torchvision import models

class ResNet50Encoder(nn.Module):
    def __init__(self, db_ind, db_attr):
        super(ResNet50Encoder, self).__init__()
        
        if db_ind == 0:
            # Add identity layer (equivalent operation in PyTorch)
            self.layer = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
            self.layer.weight.data = torch.tensor([[[[1]], [[0]], [[0]]]], dtype=torch.float32)
        else:
            # Load pre-trained model and modify it
            #REVIEW - Since we use PyTorch, suppose that we just use pretrained model from torchvision
            pretrained_model = models.resnet50(pretrained=True)
            
            # Remove layers after the specified index: layer_index = db_attr[0]
            # modified_pretrained_layers = list(pretrained_model.children())[:layer_index]
            pretrained_layers = list(pretrained_model.children())
            
            '''To see how I truncated the network, see the following link: 
            https://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.svg
            In MCNCC paper, they used 'res2bx' activations from the pre-trained network.''' 
            modified_pretrained_layers = pretrained_layers[:3]
            modified_pretrained_layers.append(pretrained_layers[4][:2])
            self.model = nn.Sequential(*modified_pretrained_layers)
            
    def forward(self, x):
        if hasattr(self, 'layer'):
            return self.layer(x)
        else:
            return self.model(x)