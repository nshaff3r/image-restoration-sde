import torch
import os
import torch.nn as nn
import cv2
import numpy as np
import torch.nn.functional as F
import subprocess
import sys
from PIL import Image
import torchvision.transforms as transforms

sys.path.append(os.path.abspath('Bringing-Old-Photos-Back-to-Life/Global/'))

from detection import detection

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, n_distributions=10, latent_dim=20):
        super(VAE, self).__init__()
        self.n_distributions = n_distributions
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute output size dynamically from input_dim
        encoder_output_dim = 256 * (input_dim[0] // 16) * (input_dim[1] // 16)
        self.fc_mu = nn.Linear(encoder_output_dim, n_distributions * latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, n_distributions * latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h).view(-1, self.n_distributions, self.latent_dim)
        logvar = self.fc_logvar(h).view(-1, self.n_distributions, self.latent_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class ScDE(nn.Module):
    def __init__(self, input_dim, filename, latent_dim=20, n_distributions=10):
        super(ScDE, self).__init__()
        self.vae = VAE(input_dim, latent_dim=latent_dim, n_distributions=n_distributions)
        self.filename = filename

    def extract_scratch_content(self, image, mask):
        return image * mask

    def forward(self, image):

        print(file)
        mask = Image.open(f"results/masks/mask/{self.filename}").convert('L')
        transform = transforms.ToTensor()
        mask_tensor = transform(mask)

        # Extract scratch content
        scratch_content = self.extract_scratch_content(image, mask_tensor)
        
        # Extract distribution using VAE
        z, mu, logvar = self.vae(scratch_content)

        normal_dists = [torch.distributions.Normal(mu[:, i, :], torch.exp(0.5 * logvar[:, i, :]))
                        for i in range(self.vae.n_distributions)]

        blur_map = custom_gaussian_blur(image)

        return z, mu, logvar, mask, blur_map


# Usage
def process_image(scde_model, image):
    z, mu, logvar, mask, blur_map = scde_model(image)
    return z, mu, logvar, mask, blur_map


def gaussian_function(x, y, sigma):
    """
    Implement the 2D Gaussian function as described in the paper.
    G(x, y) = (1 / (2π * σ^2)) * e^(-(x^2 + y^2) / (2σ^2))
    """
    return (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

def gaussian_kernel(size, sigma):
    """Generate a 2D Gaussian kernel using the explicit formula."""
    kernel = np.fromfunction(
        lambda x, y: gaussian_function(x - (size-1)/2, y - (size-1)/2, sigma),
        (size, size)
    )
    return kernel / np.sum(kernel)  # Normalize the kernel

def custom_gaussian_blur(image, kernel_size=31, sigma=100):
    """
    Apply custom Gaussian blur to a grayscale image.
    
    Args:
    image (numpy.ndarray): Input grayscale image (256x256)
    kernel_size (int): Size of the Gaussian kernel (default: 31)
    sigma (float): Standard deviation of the Gaussian distribution (default: 100)
    
    Returns:
    numpy.ndarray: Blurred image (Blur Map)
    """
    # Ensure the image is in the correct format
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    
    # Generate the Gaussian kernel using the explicit formula
    kernel = gaussian_kernel(kernel_size, sigma)
    
    # Apply convolution
    blur_map = cv2.filter2D(image, -1, kernel)
    
    return blur_map

if __name__ == "__main__":
    detection()
    for file in os.listdir("/Users/nshaff3r/Downloads/image-restoration-sde/codes/data/datasets/nypl/testH/LQ"):
        image_path = os.path.join("/Users/nshaff3r/Downloads/image-restoration-sde/codes/data/datasets/nypl/testH/LQ", file)
        image = Image.open(image_path).convert('L')
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        _, height, width = image_tensor.shape
        scde_model = ScDE(file, input_dim=(height, width))

        z, mu, logvar, mask = process_image(scde_model, image_tensor)
        
        print("Latent representation shape:", z.shape)
        print("Mean shape:", mu.shape)
        print("Log variance shape:", logvar.shape)
        print("Mask shape:", mask.shape)