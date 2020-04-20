import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class NlayerDiscriminator(nn.Module):
    def __init__(self, input_nc, nf=64, n_layers=3, norm_layer=nn.BatchNorm2d): # default norm_layer = BatchNorm2d
        super(NlayerDiscriminator, self).__init__()
        self.layers = n_layers

        kw = 4
        sw=2
        padw = int(np.ceil((kw-1.0)/2))

        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=sw, padding=padw), nn.LeakyReLU(0.2, True)]]

        for n in range(1, self.layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sw = 1 if n == self.layers - 1 else 2
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=sw, padding=padw), 
                        norm_layer(nf),
                        nn.LeakyReLU(0.2, False)]]
        
        # the last layer of the discriminator is a convolutional layer
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        sequence_stream = []
        for n in range(len(sequence)):
            sequence_stream += sequence[n]
        self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        return self.model(input)

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc, nf=64, n_layers=3, norm_layer=nn.BatchNorm2d, multi_num=3):
        super(MultiScaleDiscriminator, self).__init__()
        self.layers = n_layers
        self.multi_num = multi_num
        
        for i in range(multi_num):
            net = NlayerDiscriminator(input_nc, nf, self.layers, norm_layer)
            setattr(self, 'multi_layer' + str(i), net.model)
        
        # downsample
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def single_forward(self, model, input):
        return 

    
    def forward(self, input1):
        result = []
        input_downsampled = input1
        for i in range(self.multi_num):
            model = getattr(self, 'multi_layer' + str(self.multi_num - 1 - i))
            result.append([model(input_downsampled)])
            
            if i != (self.multi_num - 1):
                input_downsampled = self.downsample(input_downsampled)
        
        return result
        


