import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from datasets.mnist import mnist
import os
import math

def log_prior(x):
    """
    Compute the elementwise log probability of a standard Gaussian, i.e.
    log(N(x | mu=0, sigma=1)).
    """
    return torch.sum(-0.5 * x**2 - np.log(np.sqrt(2 * math.pi)), dim=1)


def sample_prior(size):
    """
    Sample from a standard Gaussian.
    """
    return torch.normal(torch.zeros(size), 1)


def get_mask():
    mask = np.zeros((28, 28), dtype='float32')
    for i in range(28):
        for j in range(28):
            if (i + j) % 2 == 0:
                mask[i, j] = 1

    mask = mask.reshape(1, 28*28)
    mask = torch.from_numpy(mask)

    return mask


class Coupling(torch.nn.Module):
    def __init__(self, c_in, mask, n_hidden=1024):
        super().__init__()
        self.n_hidden = n_hidden
        self.c_in = c_in

        # Assigns mask to self.mask and creates reference for pytorch.
        self.register_buffer('mask', mask)

        # Create shared architecture to generate both the translation and
        # scale variables.
        # Suggestion: Linear ReLU Linear ReLU Linear.
        self.nn = torch.nn.Sequential(
            nn.Linear(c_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden,c_in*2)
        )

        self.tanh = nn.Tanh()

        # The nn should be initialized such that the weights of the last layer
        # is zero, so that its initial transform is identity.
        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, z, ldj, reverse=False):
        # Implement the forward and inverse for an affine coupling layer. Split
        # the input using the mask in self.mask. Transform one part with
        # Make sure to account for the log Jacobian determinant (ldj).
        # For reference, check: Density estimation using RealNVP.
        out = self.nn(self.mask * z)
        T, S = out.split(self.c_in, dim=1)

        # NOTE: For stability, it is advised to model the scale via:
        # log_scale = tanh(h), where h is the scale-output
        # from the NN.
        S = self.tanh(S)

        if not reverse:
            z = self.mask * z + (1 - self.mask) * (z * torch.exp(S) + T)
            ldj = ldj + torch.sum((1 - self.mask) * S, dim=1)

        else:
            z = self.mask * z +(1 - self.mask) * (z - T) * torch.exp(-S)

        return z, ldj


class Flow(nn.Module):
    def __init__(self, shape, device, n_flows=4):
        super().__init__()
        channels, = shape

        mask = get_mask()

        self.layers = torch.nn.ModuleList()

        for i in range(n_flows):
            self.layers.append(Coupling(c_in=channels, mask=mask))
            self.layers.append(Coupling(c_in=channels, mask=1-mask))

        self.z_shape = (channels,)

        self.device = device
        self.to(device)

    def forward(self, z, logdet, reverse=False):
        if not reverse:
            for layer in self.layers:
                z, logdet = layer(z, logdet)
        else:
            for layer in reversed(self.layers):
                z, logdet = layer(z, logdet, reverse=True)

        return z.to(self.device), logdet.to(self.device)


class Model(nn.Module):
    def __init__(self, shape, device):
        super().__init__()
        self.flow = Flow(shape, device)

        self.device = device
        self.to(device)

    def dequantize(self, z):
        return z + torch.rand_like(z)

    def logit_normalize(self, z, logdet, reverse=False):
        """
        Inverse sigmoid normalization.
        """
        alpha = 1e-5

        if not reverse:
            # Divide by 256 and update ldj.
            z = z / 256.
            logdet -= np.log(256) * np.prod(z.size()[1:])

            # Logit normalize
            z = z*(1-alpha) + alpha*0.5
            logdet += torch.sum(-torch.log(z) - torch.log(1-z), dim=1)
            z = torch.log(z) - torch.log(1-z)

        else:
            # Inverse normalize
            logdet += torch.sum(torch.log(z) + torch.log(1-z), dim=1)
            z = torch.sigmoid(z)

            # Multiply by 256.
            z = z * 256.
            logdet += np.log(256) * np.prod(z.size()[1:])

        return z, logdet

    def forward(self, input):
        """
        Given input, encode the input to z space. Also keep track of ldj.
        """
        z = input
        ldj = torch.zeros(z.size(0), device=z.device)

        z = self.dequantize(z)
        z, ldj = self.logit_normalize(z, ldj)

        z, ldj = self.flow(z, ldj)

        # Compute log_pz and log_px per example
        log_px = log_prior(z) + ldj

        return log_px

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Sample from prior and create ldj.
        Then invert the flow and invert the logit_normalize.
        """
        z = sample_prior((n_samples,) + self.flow.z_shape).to(self.device)
        ldj = torch.zeros(z.size(0), device=z.device)

        # inverse flow
        z, ldj = self.flow.forward(z, ldj, reverse=True)
        # inverse sigmoid normalization
        z, _ = self.logit_normalize(z, ldj, reverse=True)

        return z

def epoch_iter(model, data, optimizer, device):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average bpd ("bits per dimension" which is the negative
    log_2 likelihood per dimension) averaged over the complete epoch.
    """
    total_bpd = 0
    for batch,_ in data:
        batch = batch.to(device)
        log_px = model.forward(batch)
        loss = -torch.mean(log_px)
        total_bpd += loss.item()
        if model.training:
            optimizer.zero_grad()
            loss.backward()
            # to prevent exploding gradient:
            torch.nn.utils.clip_grad_norm_(model.parameters(),10)
            optimizer.step()
    return total_bpd/(len(data)*(28**2)*np.log(2))


def run_epoch(model, data, optimizer, device):
    """
    Run a train and validation epoch and return average bpd for each.
    """
    traindata, valdata = data

    model.train()
    train_bpd = epoch_iter(model, traindata, optimizer, device)

    model.eval()
    val_bpd = epoch_iter(model, valdata, optimizer, device)

    return train_bpd, val_bpd

def save_bpd_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train bpd')
    plt.plot(val_curve, label='validation bpd')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('bpd')
    plt.tight_layout()
    plt.savefig(filename)

def save_grid_image(model, path, nrow=4, h=28):
    samples = model.sample(nrow**2).cpu().detach().numpy()
    samples = samples.reshape(-1,h,h)
    I = np.zeros((nrow*h,nrow*h))
    plt.clf()
    for i,sample in enumerate(samples):
        x_pos = i%nrow
        y_pos = i//nrow
        I[y_pos*h:y_pos*h+h,x_pos*h:x_pos*h+h] = sample
    plt.axis('off')
    plt.imshow(I.reshape(nrow*h,nrow*h), cmap="gray")
    plt.savefig(path)

def main():
    # device
    assert ARGS.device == 'cpu' or ARGS.device == 'cuda'
    device = torch.device(ARGS.device)

    data = mnist()[:2]  # ignore test split

    model = Model([784], device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        bpds = run_epoch(model, data, optimizer, device)
        train_bpd, val_bpd = bpds
        train_curve.append(train_bpd)
        val_curve.append(val_bpd)
        print("[Epoch {epoch}] train bpd: {train_bpd} val_bpd: {val_bpd}".format(
            epoch=epoch, train_bpd=train_bpd, val_bpd=val_bpd))

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functionality that is already imported.
        #  Save grid to images_nfs/
        # --------------------------------------------------------------------
        save_grid_image(model, './res/NF/grid_' + str(epoch) + '.png', nrow=10)

    save_bpd_plot(train_curve, val_curve, './res/NF/nfs_bpd.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--device', type=str, default="cpu",
                        help="Which device to use: 'cpu' or 'cuda'")
    ARGS = parser.parse_args()

    main()
