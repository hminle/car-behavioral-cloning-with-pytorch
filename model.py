import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
#from torchnet.meter import AverageValueMeter
import torch.backends.cudnn as cudnn


class CarModel(nn.Module):
    def __init__(self):
        super(CarModel, self).__init__()
        self.conv_layers = nn.Sequential(
            # input is batch_size x 3 x 66 x 200
            nn.Conv2d(3, 24, 5, stride=2, bias=False),
            #nn.ELU(0.2, inplace=True),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(36),
            
            nn.Conv2d(36, 48, 5, stride=2, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(48),
            
            nn.Conv2d(48, 64, 3, stride=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, 3, stride=1, bias=False),
            nn.ELU(),
            nn.Dropout(p=0.4)
        )
        self.linear_layers = nn.Sequential(
            #input from sequential conv layers
            nn.Linear(in_features=64*1*18, out_features=100, bias=False),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50, bias=False),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10, bias=False),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1, bias=False))
        self._initialize_weights()
        
    # custom weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, mean=1, std=0.02)
                init.constant(m.bias, 0)

    def forward(self, input):
        output = self.conv_layers(input)
        output = output.view(output.size(0), 64*1*18)
        output = self.linear_layers(output)
        return output
    
class CarDenseModel(nn.Module):
    def __init__(self):
        super(CarDenseModel, self).__init__()
        self.conv_layers = nn.Sequential(
            # input is batch_size x 3 x 66 x 200
            nn.Conv2d(3, 24, 5, stride=2, bias=False),
            #nn.ELU(0.2, inplace=True),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(36),
            
            nn.Conv2d(36, 48, 5, stride=2, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(48),
            
            nn.Conv2d(48, 64, 3, stride=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, 3, stride=1, bias=False),
            nn.ELU(),
            nn.Dropout(p=0.4)
        )
        self.linear_layers = nn.Sequential(
            #input from sequential conv layers
            nn.Linear(in_features=64*1*18+9*32*99, out_features=100, bias=False),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50, bias=False),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10, bias=False),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1, bias=False))
        
        self.shortcut = nn.Sequential(
                nn.Conv2d(3, 9, 3, stride=2, bias=False),
                nn.BatchNorm2d(9)
            )
        
        self._initialize_weights()
        
    # custom weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, mean=1, std=0.02)
                init.constant(m.bias, 0)

    def forward(self, input):
        output = self.conv_layers(input)
        
        output = output.view(output.size(0), 64*1*18)
        input_shortcut = self.shortcut(input)
        
        input_shortcut = input_shortcut.view(input_shortcut.size(0), 9*32*99)
        output_sum = torch.cat((output, input_shortcut), 1).view(output.size(0), 64*1*18+9*32*99)
        output_sum = self.linear_layers(output_sum)
        return output_sum
    
    
class CarSimpleModel(nn.Module):
    def __init__(self):
        super(CarSimpleModel, self).__init__()
        self.conv_layers = nn.Sequential(
            # input is batch_size x 3 x 16 x 32
            nn.Conv2d(3, 24, 3, stride=2, bias=False),
            nn.ELU(),
            nn.Conv2d(24, 48, 3, stride=2, bias=False),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.25)
        )
        self.linear_layers = nn.Sequential(
            #input from sequential conv layers
            nn.Linear(in_features=48*4*19, out_features=50, bias=False),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10, bias=False),
            nn.Linear(in_features=10, out_features=1, bias=False),
        )
        self._initialize_weights()
        
    # custom weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, mean=1, std=0.02)
                init.constant(m.bias, 0)

    def forward(self, input):
        
        input = input.view(input.size(0), 3, 75, 320)
        output = self.conv_layers(input)
        #print('size ' + str(output.size()))
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output    