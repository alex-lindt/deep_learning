import argparse

import torch
import torch.nn as nn
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        #layer
        self.hidden = nn.Linear(28**2, hidden_dim)
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.std = nn.Linear(hidden_dim, z_dim)

        # init weights of layers
        self.init_weights([self.hidden.weight, self.mean.weight, self.std.weight])

        self.tanh = nn.Tanh()

    def init_weights(self, weights):
        """
            Initialize the given weights.
        """
        for w in weights:
            nn.init.xavier_uniform_(w)
        return

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        h = self.tanh(self.hidden(input))
        std = torch.exp(self.std(h)) # see paper by Kingma et al.
        return self.mean(h),std

class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        # Bernouilli MLP as Decoder
        self.model =  nn.Sequential(nn.Linear(z_dim, hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, 28**2),
                                    nn.Sigmoid())

        self.model.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer)==nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)
        return

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        return self.model(input)


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def loss(self, input, output, mu, sigma_sq, e=1e-10):
        """
        Negative Average ELBO.
        """
        L_rec = - torch.sum((input*torch.log(output+e)+
                             (1-input)*torch.log(1-output)), dim=1)
        L_reg = 0.5*torch.sum(sigma_sq+mu**2-1-torch.log(sigma_sq+e), dim=1)
        return torch.mean(L_rec+L_reg, dim=0) # mean over all samples

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        mean, std = self.encoder(input)
        input_noise = torch.normal(mean=torch.zeros(self.z_dim), std=1)
        z = mean + std * input_noise
        output = self.decoder(z)
        return self.loss(input, output, mean, std)

    def sample(self, z, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        output = self.decoder(z).reshape(-1,1,28,28)
        return output, torch.mean(output, dim=0)

def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)

def save_grid_image(grid, name):
    grid = np.transpose(grid, (1, 2, 0))
    plt.axis('off')
    plt.imshow(grid, interpolation='nearest')
    plt.savefig(name)
    return

def save_manifold_image(model):
    # mesh over latent space
    x = stats.norm.ppf(np.linspace(0,1,15)[1:14])
    x_mesh = np.array([[i, j] for i in x for j in x])
    # sample images
    z = torch.from_numpy(x_mesh).float()
    images = model.decoder(z).reshape(-1,1,28,28)
    # save images as grid
    grid = make_grid(images, nrow=13).detach().numpy()
    save_grid_image(grid, './res/VAE/manifold.png')
    return

def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    losses = []
    for batch in data:
        batch = batch.reshape(-1,28**2)
        loss = model.forward(batch)
        losses.append(loss.item())
        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return np.mean(losses)

def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data[:2]

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo

def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []

    z_test = torch.normal(mean=torch.zeros((100, ARGS.zdim)), std=1)

    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Plot samples from model during training
        # --------------------------------------------------------------------
        if epoch%2==0:

            images, images_mean = model.sample(z_test, 100)

            grid = make_grid(images,nrow=10).detach().numpy()
            save_grid_image(grid,'./res/VAE/grid_' + str(epoch)+'.png')

            mean = make_grid(images_mean,nrow=1).detach().numpy()
            save_grid_image(mean,'./res/VAE/mean_' + str(epoch)+'.png')

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    if ARGS.zdim==2:
        save_manifold_image(model)

    save_elbo_plot(train_curve, val_curve, './res/VAE/elbo.pdf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=60, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
