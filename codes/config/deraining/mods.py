import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import torch.nn.functional as F
import subprocess
import sys
import torchvision.transforms as transforms
import os
import random
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms.functional as TF



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

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.ReLU(),
        )
        
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(1024 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 4 * 4, latent_dim)
        
        # Fully connected layer to decode from latent space
        self.fc_decode = nn.Linear(latent_dim, 1024 * 4 * 4)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.Sigmoid()  # Output range [0, 1]
        )
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 1024, 4, 4)  # Reshape to match the decoder's input
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# VAE Loss (Reconstruction + KL Divergence)
def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')  # Reconstruction Loss (MSE)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL Divergence Loss
    return recon_loss + kld_loss


class ScDE(nn.Module):
    def __init__(self, filename, input_dim, latent_dim=20, n_distributions=10):
        super(ScDE, self).__init__()
        self.vae = VAE(input_dim, latent_dim=latent_dim, n_distributions=n_distributions)
        self.filename = filename

    def extract_scratch_content(self, image, mask):
        print("TESTING", image.shape, mask.shape)
        return image * mask

    def forward(self, image):
        print(f"Input image shape: {image.shape}")
        
        mask = Image.open(f"results/masks/mask/{self.filename.replace('.jpg', '.png')}").convert('L')
        transform = transforms.ToTensor()
        mask_tensor = transform(mask)
        print(f"Mask shape: {mask_tensor.shape}")

        # Extract scratch content
        scratch_content = self.extract_scratch_content(image, mask_tensor)
        print(f"Scratch content shape: {scratch_content.shape}")
        
        # Extract distribution using VAE
        z, mu, logvar = self.vae(scratch_content)

        normal_dists = [torch.distributions.Normal(mu[:, i, :], torch.exp(0.5 * logvar[:, i, :]))
                        for i in range(self.vae.n_distributions)]

        blur_map = custom_gaussian_blur(image.squeeze().numpy())
        blur_map_tensor = torch.from_numpy(blur_map).unsqueeze(0)

        return z, mu, logvar, mask_tensor, blur_map_tensor

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

class ImageMaskDataset(data.Dataset):
    """
    Dataset that reads images and corresponding masks from directories,
    applies resizing and augmentations to expand the dataset by a factor of 4.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.image_paths = sorted(self._get_image_paths(opt["dataroot_images"]))
        self.mask_paths = sorted(self._get_image_paths(opt["dataroot_masks"]))
        self.resize_size = (128, 128)  # Resize to 128x128

        assert len(self.image_paths) == len(self.mask_paths), \
            "Error: Mismatch in number of images and masks."

    def _get_image_paths(self, dataroot):
        """Fetch image paths from the given directory."""
        image_paths = []
        for root, _, files in os.walk(dataroot):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __getitem__(self, index):
        """
        Get an image and its corresponding mask, apply resizing to 128x128,
        and apply one of four augmentations based on the index.
        The dataset size is artificially increased by 4 via augmentations.
        """
        original_index = index // 4  # Get the original image index
        aug_type = index % 4  # Determine which augmentation to apply

        # Fetch the image and mask
        image_path = self.image_paths[original_index]
        mask_path = self.mask_paths[original_index]

        img = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Assuming mask is grayscale

        # Resize images and masks
        img = img.resize(self.resize_size, Image.BILINEAR)
        mask = mask.resize(self.resize_size, Image.NEAREST)  # Use NEAREST for masks

        # Apply augmentations
        img, mask = self.apply_augmentations(img, mask, aug_type)

        # Convert images and masks to tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)

        return {"image": img, "mask": mask, "image_path": image_path, "mask_path": mask_path}

    def apply_augmentations(self, img, mask, aug_type):
        """
        Apply one of four augmentations: 0 = original, 1 = horizontal flip,
        2 = vertical flip, 3 = 90-degree rotation.
        """
        if aug_type == 1:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        elif aug_type == 2:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        elif aug_type == 3:
            img = TF.rotate(img, 90)
            mask = TF.rotate(mask, 90)
        # aug_type == 0 is the original image, no changes

        return img, mask

    def __len__(self):
        # The dataset size is multiplied by 4 due to augmentations
        return len(self.image_paths) * 4


        
if __name__ == "__main__":
    detection()

    # opt = {
    #     "dataroot_images": "/path/to/images",
    #     "dataroot_masks": "/path/to/masks"
    # }
    # dataset = ImageMaskDataset(opt)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)



    # vae = VAE(in_channels=1, latent_dim=128).cuda()
    # optimizer = optim.Adam(vae.parameters(), lr=1e-4)

    # def train_vae(dataloader, num_epochs=10):
    #     vae.train()
    #     for epoch in range(num_epochs):
    #         train_loss = 0
    #         for images in dataloader:
    #             images = images.cuda()

    #             optimizer.zero_grad()

    #             # Forward pass
    #             recon_images, mu, logvar = vae(images)
    #             loss = vae_loss_function(recon_images, images, mu, logvar)

    #             # Backward pass and optimization
    #             loss.backward()
    #             optimizer.step()

    #             train_loss += loss.item()

    #         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(dataloader.dataset):.4f}')

    # # detection()
    # for file in os.listdir("/home/nolanshaffer/slurm/image-restoration-sde/codes/config/deraining/results/masks/input"):
    #     image_path = os.path.join("/home/nolanshaffer/slurm/image-restoration-sde/codes/config/deraining/results/masks/input", file)
    #     image = Image.open(image_path).convert('L')
    #     transform = transforms.ToTensor()
    #     image_tensor = transform(image)
    #     _, height, width = image_tensor.shape
    #     scde_model = ScDE(file, (height, width))

    #     z, mu, logvar, mask = process_image(scde_model, image_tensor)
        
    #     print("Latent representation shape:", z.shape)
    #     print("Mean shape:", mu.shape)
    #     print("Log variance shape:", logvar.shape)
    #     print("Mask shape:", mask.shape)
