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
epochs = 100 #30 # number of epoch to train
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
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz # the input noise vector

        # the rest of the codes to be filled ...

        # defining the convolution block used 
        def block(in_feat, out_feat, norm=True, stride=2, padding=1):
            if norm:
                return nn.Sequential(
                    nn.ConvTranspose2d(in_feat, out_feat, kernel_size=4, stride=stride, padding=padding, bias=False),
                    nn.BatchNorm2d(out_feat),  # Normalizes activations
                    nn.ReLU(inplace=True)
                )
            else:
                return nn.Sequential(
                    nn.ConvTranspose2d(in_feat, out_feat, kernel_size=4, stride=stride, padding=padding, bias=False),
                    nn.ReLU(inplace=True)
                )
            
        self.model = nn.Sequential(
            block(self.nz, 128, norm=False, stride=1, padding=0), 
            block(128, 64),
            block(64, 32),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()   #output (-1,1) values for the generated image
        )

    def forward(self, noise_input):
        return self.model(noise_input)

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
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0, bias=False),  #here its flattened
            nn.Sigmoid() #chooses fake or real guess
        )

    def forward(self, image):
        return self.model(image) 

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
        save_image(generated_img, f"outputs/gen_img{epoch}.png", normalize=True)
        epoch_loss_g = loss_g / batch_idx # total generator loss for the epoch
        epoch_loss_d = loss_d / batch_idx # total discriminator loss for the epoch
        losses_g.append(epoch_loss_g.cpu().detach().numpy())
        losses_d.append(epoch_loss_d.cpu().detach().numpy())
        print(f"Epoch {epoch+1} of {epochs}")
        print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")


    # create the plots of the losses

    import matplotlib.pyplot as plt

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(9,6))
    plt.plot(range(epochs), losses_g, label="generator")
    plt.plot(range(epochs), losses_d, label="discriminator")
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("outputs/loss.png")
