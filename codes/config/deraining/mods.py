import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import sys


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
    def __init__(self, input_dim=256, hidden_dim=512, latent_dim=20):
        super(VAE, self).__init__()
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
        
        self.fc_mu = nn.Linear(256 * (input_dim // 16) * (input_dim // 16), latent_dim)
        self.fc_logvar = nn.Linear(256 * (input_dim // 16) * (input_dim // 16), latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class ScDE(nn.Module):
    def __init__(self, input_dim=256, latent_dim=20):
        super(ScDE, self).__init__()
        self.vae = VAE(input_dim, latent_dim=latent_dim)

    def extract_scratch_content(self, image, mask):
        return image * mask

    def forward(self, image):
        args = [
                "--test_path /Users/nshaff3r/Downloads/image-restoration-sde/codes/data/datasets/nypl/testH/LQ", 
                "--output_dir /Users/nshaff3r/Downloads/image-restoration-sde/codes/config/deraining/results/masks",
                "--input_size full_size"
                ]
        command = ["python3", "Bringing-Old-Photos-Back-to-Life/Global/detection.py"] + args
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Mask output:", result.stdout)

        

        # Extract scratch content
        scratch_content = self.extract_scratch_content(image, mask)
        
        # Extract distribution using VAE
        z, mu, logvar = self.vae(scratch_content)
        
        return z, mu, logvar, mask

# Usage
def process_image(scde_model, image):
    z, mu, logvar, mask = scde_model(image)
    
    # mu and logvar represent the multidimensional μ and σ
    # z is the latent representation of the scratch distribution
    
    return z, mu, logvar, mask

# Initialize the model
scde_model = ScDE()

# Process an image (assuming 'image' is a 256x256 grayscale image tensor)
image = torch.randn(1, 1, 256, 256)  # Example input
z, mu, logvar, mask = process_image(scde_model, image)

print("Latent representation shape:", z.shape)
print("Mean shape:", mu.shape)
print("Log variance shape:", logvar.shape)
print("Mask shape:", mask.shape)
