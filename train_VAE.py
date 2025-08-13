import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from models import VAE  # uses your VAE class


def get_dataloaders(batch_size, data_dir="./data"):
    tfm = transforms.Compose([
        transforms.ToTensor(),  # [0,1]
    ])
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=tfm)
    val_ds   = datasets.MNIST(root=data_dir, train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


@torch.no_grad()
def plot_latent_space(model, loader, device, max_points=10000, save_path=None):
    """Return a Matplotlib figure of the 2D latent space colored by labels."""
    model.eval()
    zs, ys = [], []
    collected = 0
    for x, y in loader:
        x = x.to(device).view(x.size(0), -1)
        mu, logvar = model.encode(x)
        z = mu  # For visualization, use mean of posterior
        zs.append(z.detach().cpu())
        ys.append(y.detach().cpu())
        collected += x.size(0)
        if collected >= max_points:
            break

    Z = torch.cat(zs, dim=0)
    Y = torch.cat(ys, dim=0)
    if Z.size(1) != 2:
        raise ValueError(f"Latent space visualization requires latent_dim=2, got {Z.size(1)}")

    fig = plt.figure(figsize=(6, 5), dpi=120)
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=Y.numpy(), s=6, alpha=0.7, cmap="tab10")
    plt.title("VAE Latent Space (2D)")
    plt.xlabel("z1"); plt.ylabel("z2")
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label("Digit label")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig


@torch.no_grad()
def plot_reconstructions(model, loader, device, save_path=None, num_images=8):
    """Save a grid of input and reconstructed images."""
    model.eval()
    x, _ = next(iter(loader))
    x = x[:num_images].to(device)
    x_flat = x.view(x.size(0), -1)
    recon, _, _ = model(x_flat)
    recon = recon.view(-1, 1, 28, 28).cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.5, 3))
    for i in range(num_images):
        axes[0, i].imshow(x[i, 0].cpu(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i, 0], cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Input", fontsize=12)
    axes[1, 0].set_ylabel("Recon", fontsize=12)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig


def vae_loss(recon, x, mu, logvar):
    # recon: [B, 784], x: [B, 784], both in [0,1]
    # mu, logvar: [B, latent_dim]
    # Reconstruction loss
    BCE = nn.functional.binary_cross_entropy(recon, x, reduction="sum")
    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (BCE + KLD) / x.size(0)  # mean per batch


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_kld = 0.0
    for x, _ in loader:
        x = x.to(device).view(x.size(0), -1)
        optimizer.zero_grad(set_to_none=True)
        recon, mu, logvar = model(x)
        BCE = nn.functional.binary_cross_entropy(recon, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = (BCE + KLD) / x.size(0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_bce += BCE.item()
        total_kld += KLD.item()
    n = len(loader.dataset)
    return total_loss / n, total_bce / n, total_kld / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_kld = 0.0
    for x, _ in loader:
        x = x.to(device).view(x.size(0), -1)
        recon, mu, logvar = model(x)
        BCE = nn.functional.binary_cross_entropy(recon, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = (BCE + KLD) / x.size(0)
        total_loss += loss.item() * x.size(0)
        total_bce += BCE.item()
        total_kld += KLD.item()
    n = len(loader.dataset)
    return total_loss / n, total_bce / n, total_kld / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Repro
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader = get_dataloaders(args.batch_size, args.data_dir)

    # Model
    input_dim = 28 * 28  # MNIST
    model = VAE(input_dim=input_dim, latent_dim=args.latent_dim).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_bce, train_kld = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_bce, val_kld = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

        # Every 5 epochs: visualize latent space (requires latent_dim=2)
        if epoch % 5 == 0 and args.latent_dim == 2:
            latent_path = os.path.join(args.save_dir, f"latent_space_epoch_{epoch:03d}.png")
            fig = plot_latent_space(model, val_loader, device, save_path=latent_path)
            plt.close(fig)

        # Every 5 epochs: save input and reconstructed images
        if epoch % 5 == 0:
            recon_path = os.path.join(args.save_dir, f"reconstructions_epoch_{epoch:03d}.png")
            fig = plot_reconstructions(model, val_loader, device, save_path=recon_path, num_images=8)
            plt.close(fig)


if __name__ == "__main__":
    main()