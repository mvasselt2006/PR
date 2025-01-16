import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np

from models import *
from utils import progress_bar
from utils import get_output

import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

#Load weights
checkpoint = torch.load(f'./checkpoint/14167203_ckpt.pth',map_location=torch.device('cpu'))
#Load the weights
net.load_state_dict(checkpoint['net'],strict=False)
best_acc = checkpoint['acc']
best_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=2) #Batch size 1 to only look at 1 image

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


## adapted from https://www.geeksforgeeks.org/visualizing-feature-maps-using-pytorch/ 
# and https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573


if __name__=='__main__':
    # Extract convolutional layers and their weights
    conv_weights = []  # List to store convolutional layer weights
    conv_layers = []  # List to store convolutional layers
    total_conv_layers = 0  # Counter for total convolutional layers

    # Traverse through the model to extract convolutional layers and their weights
    for module in net.children():
        if isinstance(module, nn.Conv2d):
            total_conv_layers += 1
            conv_weights.append(module.weight)
            conv_layers.append(module)
        elif isinstance(module,nn.Sequential):
            for j in range(len(module)):
                for child in module[j].children():
                    if type(child) == nn.Conv2d:
                        total_conv_layers+=1
                        conv_weights.append(child.weight)
                        conv_layers.append(child)

    print(f'Total convolution layers: {total_conv_layers}')

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if batch_idx>0:
                break   #Stop after first batch
            inputs, targets = inputs.to(device), targets.to(device)

            #Get original image
            input_img= inputs
            input_img=input_img.squeeze(0)
            plot_image= input_img.data.cpu().numpy()
            #change channels to have colour last
            im2display = plot_image.transpose((1,2,0))

            fig=plt.figure(); ax=fig.subplots()
            ax.imshow(im2display); ax.set_title(classes[targets]); ax.axis("off")
            fig.savefig("input_image.png")
            plt.show();plt.close()




            output_size=[]
            feature_maps = []  # List to store feature maps
            layer_names = []  # List to store layer names

            layer_numbers= [0,11,len(conv_layers)-1] #Layer number for which we save the featuremaps

            #Get sizes after each layer
            for i,layer in enumerate(conv_layers):
                inputs = layer(inputs)
                output_size.append(inputs.shape)
                if i in layer_numbers:
                    feature_maps.append(inputs)
                    layer_names.append(str(layer))
    

            # # Extract feature maps
            # feature_maps = []  # List to store feature maps
            # layer_names = []  # List to store layer names
            # for layer in conv_layers[0,10,-1]: #Chose first, last and random layer (out of 17)
            #     input_image = layer(inputs)
            #     feature_maps.append(inputs)
            #     layer_names.append(str(layer))

            # Process and visualize feature maps
            processed_feature_maps = []  # List to store processed feature maps
            for feature_map in feature_maps:
                feature_map = feature_map.squeeze(0)  # Remove the batch dimension
                mean_feature_map = torch.sum(feature_map, 0) / feature_map.shape[0]  # Compute mean across channels
                processed_feature_maps.append(mean_feature_map.data.cpu().numpy()) #Append the mean of the channels
                #Also chose two channels, we will take the first and the last
                processed_feature_maps.append(feature_map[0,:,:].data.cpu().numpy())
                processed_feature_maps.append(feature_map[-1,:,:].data.cpu().numpy())

            # Plot the feature maps
            fig = plt.figure(figsize=(10, 10))
            channel_numbers=["mean", "first", "last"]
            for i in range(len(processed_feature_maps)):
                ax = fig.add_subplot(3, 3, i + 1)
                ax.imshow(processed_feature_maps[i])
                ax.axis("off")
                layer_name=layer_names[i//3].split('(')[0]
                ax.set_title(f'{layer_name}:{layer_numbers[i//3]+1},{channel_numbers[i%3]} channel', fontsize='small')
            fig.savefig("Featuremap.png")
            plt.show(); plt.close()
            
            for size in output_size:

                print(size,'\n')
