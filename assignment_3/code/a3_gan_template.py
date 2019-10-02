import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

def noisy_labels(shape, i=1):
    assert i==1 or i==0
    noise =np.random.uniform(0,0.3,shape)
    if i==1:
        return torch.ones(shape) - torch.from_numpy(noise).float()
    return torch.zeros(shape) + torch.from_numpy(noise).float()

class Generator(nn.Module):
    def __init__(self, device):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(args.latent_dim, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 784),
            nn.Tanh())

        self.to(device)

    def forward(self, z):
        # Generate images from z
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, device):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784,512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512,256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

        self.to(device)

    def forward(self, img):
        # return discriminator score for img
        return self.model(img)

def save_loss_plot(dis_loss, gen_loss, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(dis_loss, label='Discriminator')
    plt.plot(gen_loss, label='Generator')
    plt.legend()
    plt.xlabel('EPOCHS')
    plt.ylabel('LOSS')
    plt.tight_layout()
    plt.savefig(filename)

def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device):

    # loss functions
    loss_funct_gen = nn.BCELoss(reduction='mean')
    loss_funct_dis = nn.BCELoss(reduction='mean')

    L_G = []
    L_D = []

    for epoch in range(args.n_epochs):

        L_G_epoch = []
        L_D_epoch = []

        for i, (imgs, _) in enumerate(dataloader):

            if imgs.shape[0] != args.batch_size:
                continue

            imgs = imgs.to(device)

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()

            z = torch.normal(mean=torch.zeros((args.batch_size, args.latent_dim)), std=1).to(device)
            G_z = generator.forward(z).to(device)
            D_G_z = discriminator.forward(G_z)

            V_G = loss_funct_gen(D_G_z, noisy_labels((args.batch_size,1),i=1).to(device))

            V_G.backward(retain_graph=True)
            optimizer_G.step()

            L_G_epoch.append(V_G.item())


            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()

            j = int(args.batch_size/2)
            input_dis = torch.cat((G_z[:j], imgs.reshape(args.batch_size, -1)[:j]), 0)
            output_true = torch.cat((noisy_labels((j,1),i=0).to(device),
                                     noisy_labels((j,1),i=1).to(device)), 0)

            output_dis = discriminator.forward(input_dis)

            V_D = loss_funct_dis(output_dis, output_true)
            V_D.backward()
            optimizer_D.step()

            L_D_epoch.append(V_D.item())

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                G_z = G_z.reshape(-1,1,28,28)

                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(G_z[:64],'./res/GAN/{}.png'.format(batches_done), nrow=8, normalize=True)

        lg = np.mean(L_G_epoch)
        L_G.append(lg)

        ld = np.mean(L_D_epoch)
        L_D.append(ld)

        # see how the loss is doing
        save_loss_plot(L_D, L_G, "./res/GAN/loss_plot.png")

        print(f"[Epoch {epoch}] Loss Generator: {lg} Loss Discriminator: {ld}")


def main():
    # device
    assert args.device=='cpu' or args.device=='cuda'
    device = torch.device(args.device)

    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),(0.5,))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(device)
    discriminator = Discriminator(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--device', type=str, default="cpu",
                        help="Which device to use: 'cpu' or 'cuda'")
    args = parser.parse_args()

    main()
