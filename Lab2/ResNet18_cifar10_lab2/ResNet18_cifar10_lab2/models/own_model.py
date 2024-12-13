import torch
import torch.nn as nn
import torch.nn.functional as F


class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5,stride=1,padding=2,bias=False)
        self.conv2= nn.Conv2d(64,128, kernel_size=3,stride=1,padding=1,bias=True)
        self.conv3= nn.Conv2d(128,256,kernel_size=3,stride=1,bias=False)
        self.conv4= nn.Conv2d(256,512, kernel_size=1,stride=1,bias=True)

        self.mxpool = nn.MaxPool2d(2, 2)

        #Batch normalisation
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512 * 7* 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x))) #32x32
        x = self.mxpool(F.leaky_relu(self.conv2(x))) #16x16

        x = F.leaky_relu(self.conv3(x)) #14x14
        x = self.mxpool(F.leaky_relu(self.bn2(self.conv4(x)))) #7x7


        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def Mod1():
    return Net1()
