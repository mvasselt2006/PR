import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision.utils import save_image
import torch.nn.functional as F
import numpy as np

# hyper-parameters configurations
image_size = 32 # we need to resize image to 32X32
batch_size = 128
nz = 100 # latent vector size
beta1 = 0.5 # beta1 value for Adam optimizer
lr = 0.0001 # learning rate  # in paper 0.0002 is used
sample_size = 32 # fixed sample size
epochs = 100 # number of epoch to train
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# image transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5)),
])

# dataset
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)


# generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        #Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5,stride=1,padding=2,bias=False)
        self.conv2= nn.Conv2d(64,128, kernel_size=3,stride=1,padding=1,bias=True)
        self.conv3= nn.Conv2d(128,256,kernel_size=3,stride=1,bias=False)
        self.conv4= nn.Conv2d(256,512, kernel_size=1,stride=1,bias=True)

        #Max pooling
        self.mxpool = nn.MaxPool2d(2, 2)

        #Batch normalisation
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(512)

        #Linear operations to reduces to final classification
        self.fc1 = nn.Linear(512 * 7* 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x))) #32x32 output
        x = self.mxpool(F.leaky_relu(self.conv2(x))) #16x16 output

        x = F.leaky_relu(self.conv3(x)) #14x14 output
        x = self.mxpool(F.leaky_relu(self.bn2(self.conv4(x)))) #7x7 output


        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x)) #Reduce to float
        return x

# discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # the rest of the codes to be filled ...
        def block(in_feat, out_feat, norm=True):
                if norm:
                    return nn.Sequential(
                        nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(out_feat),  # Normalizes activations
                        nn.LeakyReLU(inplace=True)
                    )
                else:
                    return nn.Sequential(
                        nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1, bias=False),
                        nn.LeakyReLU(inplace=True)
                    )

            
        self.model = nn.Sequential(
            block(3, 32, norm=False),
            block(32, 64),
            block(64, 128),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0, bias=False),  #have to look into why stride and padding this way
            nn.Sigmoid()
        )

    def forward(self, image):
        return self.model(image)   #see some things about flattening the image, might be needed here


# train the network

# initialize models
generator = Generator(nz).to(device)
discriminator = Discriminator().to(device)

# optimizers
optim_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optim_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# loss function
criterion = nn.BCELoss()

# to store the generator and discriminator loss after each epoch
losses_g = []
losses_d = []

# creat the noise signal as input to the generator
def create_noise(sample_size,nz):
  return torch.randn(sample_size,nz,1,1).to(device)

# function to train the discriminator network
def train_discriminator(optimizer, data_real, data_fake):
    b_size = data_real.size(0)
    # get the real label vector
    real_label = torch.ones(b_size, 1).to(device).squeeze()
    # get the fake label vector
    fake_label = torch.zeros(b_size, 1).to(device).squeeze()
    optimizer.zero_grad()
    # get the outputs by doing real data forward pass
    output_real = discriminator(data_real).view(-1)
    #print(output_real.size())
    loss_real = criterion(output_real, real_label)
    # get the outputs by doing fake data forward pass
    output_fake = discriminator(data_fake).view(-1)
    loss_fake = criterion(output_fake, fake_label)
    # compute gradients of real loss
    loss_real.backward()
    # compute gradients of fake loss
    loss_fake.backward()
    # update discriminator parameters
    optimizer.step()
    return loss_real + loss_fake


# function to train the generator network
def train_generator(optimizer, data_fake):
    b_size = data_fake.size(0)
    # get the real label vector
    real_label = torch.ones(b_size, 1).to(device).squeeze()
    optimizer.zero_grad()
    # output by doing a forward pass of the fake data through discriminator
    output = discriminator(data_fake).view(-1)
    loss = criterion(output, real_label)
    # compute gradients of loss
    loss.backward()
    # update generator parameters
    optimizer.step()
    return loss

# create the noise vector
if __name__ == "__main__":
    noise = create_noise(sample_size, nz)

    for epoch in range(epochs):
        loss_g = 0.0
        loss_d = 0.0
        for batch_idx, data in enumerate(trainloader):
            image, _ = data
            image = image.to(device)
            b_size = len(image)
            # forward pass through generator to create fake data
            data_fake = generator(create_noise(b_size, nz)).detach()
            data_real = image
            loss_d += train_discriminator(optim_d, data_real, data_fake)
            data_fake = generator(create_noise(b_size, nz))
            loss_g += train_generator(optim_g, data_fake)
        # final forward pass through generator to create fake data...
        # ...after training for current epoch
        generated_img = generator(noise).cpu().detach()
        # save the generated torch tensor models to disk
        save_image(generated_img, f"outputs/gen_img{epoch}.png", normalize=True)           # so in this folder the images can be viewed during running
        epoch_loss_g = loss_g / batch_idx # total generator loss for the epoch
        epoch_loss_d = loss_d / batch_idx # total discriminator loss for the epoch
        losses_g.append(epoch_loss_g.cpu().detach().numpy())
        losses_d.append(epoch_loss_d.cpu().detach().numpy())
        print(f"Epoch {epoch+1} of {epochs}")
        print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")


    #now the plotting of losses, dont know what is smart coding wise. 

    import matplotlib.pyplot as plt

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(9,6))
    plt.plot(range(epochs), losses_g, label="generator")
    plt.plot(range(epochs), losses_d, label="discriminator")
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.savefig("outputs/loss.png")
