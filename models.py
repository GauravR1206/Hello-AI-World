import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VAE(nn.Module):
    """This is a VAE model for MNIST dataset. If using other datasets, you need to change the input_dim"""
    def __init__(self, input_dim=784, latent_dim=2):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc21 = nn.Linear(128, latent_dim)  # mu
        self.fc22 = nn.Linear(128, latent_dim)  # logvar
        # Decoder
        self.fc4 = nn.Linear(latent_dim, 128)
        self.fc5 = nn.Linear(128, 256)
        self.fc6 = nn.Linear(256, 512)
        self.fc7 = nn.Linear(512, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return self.fc21(h3), self.fc22(h3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        h6 = F.relu(self.fc6(h5))
        return torch.sigmoid(self.fc7(h6))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

class AE(nn.Module):
    """This is a regular Autoencoder model for MNIST dataset. If using other datasets, you need to change the input_dim"""
    def __init__(self, input_dim=784, latent_dim=2):
        super(AE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, latent_dim)
        # Decoder
        self.fc5 = nn.Linear(latent_dim, 128)
        self.fc6 = nn.Linear(128, 256)
        self.fc7 = nn.Linear(256, 512)
        self.fc8 = nn.Linear(512, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        z = self.fc4(h3)
        return z

    def decode(self, z):
        h4 = F.relu(self.fc5(z))
        h5 = F.relu(self.fc6(h4))
        h6 = F.relu(self.fc7(h5))
        return torch.sigmoid(self.fc8(h6))

    def forward(self, x):
        z = self.encode(x.view(-1, 784))
        recon_x = self.decode(z)
        return recon_x, z