import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)  # MNIST images are 28x28

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        return self.fc(x)
    
    def __str__(self):
        return "SimpleModel"

class myResNet(nn.Module):
    def __init__(self):
        super(myResNet, self).__init__()
        ## COMPLETED ##
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.active1 = nn.ReLU()

        # Residual Unit
        self.layer2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding="same"
        )
        self.active2 = nn.ReLU()
        self.layer2_5 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding="same"
        )  # layer 2.5

        # Residual Unit end
        self.active_residual = nn.ReLU()

        ##### Change up until this point.
        self.PL3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.active4 = nn.ReLU()

        self.layer5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.active5 = nn.ReLU()

        self.PL6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer7 = nn.Linear(in_features=1024, out_features=128)
        self.active7 = nn.ReLU()

        self.layer8 = nn.Linear(in_features=128, out_features=10)
        self.active8 = nn.Sigmoid()

    def forward(self, x):
        x_dim = x.dim()
        x = self.active1(self.layer1(x))

        x1 = x  # store for skip connection

        # Residual Unit
        x = self.layer2(x)
        x = self.active2(x)
        x = self.layer2_5(x)
        x2 = x  # store for skip connection
        # Residual Unit end
        x = x1 + x2
        x = self.active_residual(x)
        ##### Change up until this point.

        x = self.PL3(x)

        x = self.active4(self.layer4(x))
        x = self.active5(self.layer5(x))
        x = self.PL6(x)

        # x = torch.flatten(x)
        # Ask ChatGPT: if input dim is 3, flat to 1D, if input dim is 4, eg, [100, xxx,xxx,xxx], flatten to [100, something]
        # ChatGPT generated code
        if x_dim == 4:  # For 4D input, flatten to [batch_size, -1]
            x = torch.flatten(x, start_dim=1)
        elif x_dim == 3:  # For 3D input, flatten to 1D
            x = torch.flatten(x)
        # ChatGPT Done

        x = self.active7(self.layer7(x))
        x = self.active8(self.layer8(x))
        return x  ## COMPLETED ##

    def __str__(self) -> str:
        return "myResNet"