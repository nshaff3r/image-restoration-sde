import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import torch.nn.functional as F
import subprocess
import sys
import torchvision.transforms as transforms
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DPP
import torch.distributed as dist
import os
import random
from torch.optim.lr_scheduler import StepLR
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms.functional as TF



sys.path.append(os.path.abspath('Bringing-Old-Photos-Back-to-Life/Global/'))

# from detection import detection

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
    def __init__(self, in_channels=3, latent_dim=128, image_height=128, image_width=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_height = image_height
        self.image_width = image_width
        
        # Encoder with more layers and increased filters
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Additional layers for larger feature maps
        )
        
        # Calculate output size of final convolution layer
        # Example calculation: (image_size - kernel_size + 2*padding) / stride + 1
        # This will give you the final feature map size after all downsampling
        self.final_feature_size = 2048 * (self.image_height // 64) * (self.image_width // 64)
        
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(self.final_feature_size, latent_dim)
        self.fc_logvar = nn.Linear(self.final_feature_size, latent_dim)
        
        # Fully connected layer to decode from latent space
        self.fc_decode = nn.Linear(latent_dim, self.final_feature_size)
        
        # Decoder with more layers and larger upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),
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
        h = h.view(h.size(0), 2048, self.image_height // 64, self.image_width // 64)  # Adjust dimensions
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# VAE Loss (Reconstruction + KL Divergence)
def vae_loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss: Mean Squared Error (MSE)
    # Assuming the images are scaled between -1 and 1
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / x.size(0)  # Normalize by batch size
    beta = 1 
    # Total loss is the sum of the reconstruction loss and KL divergence loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss

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

        img = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')  # Assuming mask is grayscale

        # Resize images and masks
        img = img.resize(self.resize_size, Image.BILINEAR)
        mask = mask.resize(self.resize_size, Image.NEAREST)  # Use NEAREST for masks

        # Apply augmentations
        img, mask = self.apply_augmentations(img, mask, aug_type)

        # Convert images and masks to tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)

        img = img * mask

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

def save_reconstruction(model, dataloader, save_dir="reconstructions"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].cuda()

            # Forward pass through VAE
            recon_images, _, _ = model(images)

            # Move to CPU and convert to numpy for saving
            images = images.cpu().numpy()
            recon_images = recon_images.cpu().numpy()

            num_images = images.shape[0]  # Number of images in the batch

            for i in range(num_images):
                # Convert original image from [-1, 1] to [0, 255]
                orig_img = ((images[i][0] * 0.5) + 0.5) * 255
                orig_img = orig_img.astype(np.uint8)

                # Convert reconstructed image from [-1, 1] to [0, 255]
                recon_img = ((recon_images[i][0] * 0.5) + 0.5) * 255
                recon_img = recon_img.astype(np.uint8)

                # Save original and reconstructed images
                orig_image_path = os.path.join(save_dir, f"original_{batch_idx}_{i}.png")
                recon_image_path = os.path.join(save_dir, f"reconstructed_{batch_idx}_{i}.png")
                
                # Save using PIL
                Image.fromarray(orig_img).save(orig_image_path)
                Image.fromarray(recon_img).save(recon_image_path)

                print(f"Saved {orig_image_path} and {recon_image_path}")


# Define the path where you want to save/load the model weights
MODEL_SAVE_PATH = 'vae_weights.pth'

def load_model_weights(model, file_path):
    """Load model weights from a file if it exists."""
    if os.path.isfile(file_path):
        print(f"Loading model weights from {file_path}...")
        model.load_state_dict(torch.load(file_path))
    else:
        print(f"No model weights found at {file_path}. Initializing model from scratch.")

def save_model_weights(model, file_path):
    """Save model weights to a file."""
    print(f"Saving model weights to {file_path}...")
    torch.save(model.state_dict(), file_path)

def validate_vae(dataloader, model, loss_function):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].cuda()
            recon_images, mu, logvar = model(images)
            loss = loss_function(recon_images, images, mu, logvar)
            val_loss += loss
    avg_val_loss = val_loss / len(dataloader)
    return avg_val_loss        

def setup_distributed():
    # Initialize the process group
    this_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    dist.init_process_group(backend='nccl', rank=this_rank, world_size=world_size)
    
    # Set the GPU device
    torch.cuda.set_device(this_rank)
    print(f"Initialized distributed training on GPU {this_rank}")


if __name__ == "__main__":
    # setup_distributed()
    opt = {
        "dataroot_images": "results/masks/input",
        "dataroot_masks": "results/masks/mask"
    }
    dataset = ImageMaskDataset(opt)
    print("LENGTH: ", len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    vae = VAE(in_channels=1, latent_dim=128).cuda()
    # vae = DPP(vae)
    optimizer = optim.Adam(vae.parameters(), lr=1e-4)
 
    load_model_weights(vae, MODEL_SAVE_PATH)
    avg_val_loss = validate_vae(dataloader, vae, vae_loss_function)
    print("VAL_LOSS: ", avg_val_loss)
    save_reconstruction(vae, dataloader)
    sys.exit()
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    def train_vae(dataloader, num_epochs=200):
        vae.train()
        for epoch in range(num_epochs):
            train_loss = 0
            for batch in dataloader:
                images = batch['image'].cuda()  # Access 'image' from the dictionary
                # Optionally, you might also need to handle masks here
                # masks = batch['mask'].cuda()
    
                optimizer.zero_grad()
    
                # Forward pass
                recon_images, mu, logvar = vae(images)
                loss = vae_loss_function(recon_images, images, mu, logvar)
    
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
    
                train_loss += loss
            scheduler.step()
            if epoch % 100 == 0:
                 print("SAVING WEIGHTS: ", epoch)
                 save_model_weights(vae, f"vae_weights_{epoch}.pth")

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(dataloader.dataset):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
    train_vae(dataloader, num_epochs=500)
    print("DONE")
    save_model_weights(vae, MODEL_SAVE_PATH)
    print("Weights saved.")
    dist.destroy_process_group()
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
