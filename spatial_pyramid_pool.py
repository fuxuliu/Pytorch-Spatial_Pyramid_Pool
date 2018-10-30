import torch
import torch.nn as nn
import numpy as np

class Spatial_Pyramid_Pool(nn.Module):
    def __init__(self, output_size, pooling_type='max'):
        '''
        output_size: the height and width after spp layer
        pooling_type: the type of pooling 
        '''
        super(Spatial_Pyramid_Pool, self).__init__()
        self.output_size = output_size
        self.pooling_type = pooling_type
    def forward(self, x):
        N, C, H, W = x.size()
        ## the size of pooling window
        sizeX  = int(np.ceil(H / self.output_size))
        ## the strides of pooling
        stride = int(np.floor(H / self.output_size))
        if self.pooling_type == 'max':
            self.spp = nn.MaxPool2d(kernel_size=sizeX, stride=stride)
        else:
            self.spp = nn.AdaptiveAvgPool2d(kernel_size=sizeX, stride=stride)
        x = self.spp(x)
        
        return x