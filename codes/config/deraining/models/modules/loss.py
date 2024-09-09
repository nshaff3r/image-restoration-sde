import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        self.layers = ['3', '8', '15', '22', '29']  # Corresponding to conv1_2, conv2_2, conv3_3, conv4_3, conv5_3
    
    def forward(self, x):
        features = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layers:
                features.append(x)
        return features

class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.vgg = VGG16FeatureExtractor()
        
        # Coefficients for pyramid losses
        self.lambda_pyr1 = 1
        self.lambda_pyr2 = 1
        self.lambda_pyr3 = 1
        self.lambda_adv = 0.2
        
        # Coefficients for individual loss components
        self.lambda_rec = 1
        self.lambda_per = 0.2
        self.lambda_sty = 250

    def forward(self, pred, target, discriminator=None):
        # Compute pyramid losses
        l_pyr1 = self.pyramid_loss(pred, target, level=1)
        l_pyr2 = self.pyramid_loss(pred, target, level=2)
        l_pyr3 = self.pyramid_loss(pred, target, level=3)
        
        # Compute adversarial loss
        l_adv = torch.tensor(0.0, device=pred.device)
        if discriminator is not None:
            l_adv = self.adversarial_loss(pred, target, discriminator)
        
        # Compute total loss
        total_loss = (
            self.lambda_pyr1 * l_pyr1 +
            self.lambda_pyr2 * l_pyr2 +
            self.lambda_pyr3 * l_pyr3 +
            self.lambda_adv * l_adv
        )
        
        return total_loss

    def pyramid_loss(self, pred, target, level):
        # Downsample images based on pyramid level
        for _ in range(level - 1):
            pred = F.avg_pool2d(pred, 2)
            target = F.avg_pool2d(target, 2)
        
        # Compute individual loss components
        l_rec = F.l1_loss(pred, target)
        l_per = self.perceptual_loss(pred, target)
        l_sty = self.style_loss(pred, target)
        
        # Combine losses
        return self.lambda_rec * l_rec + self.lambda_per * l_per + self.lambda_sty * l_sty

    def replicate_grayscale(self, x):
        return x.repeat(1, 3, 1, 1) if x.size(1) == 1 else x

    def perceptual_loss(self, pred, target):
        pred = self.replicate_grayscale(pred)
        target = self.replicate_grayscale(target)
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return sum(F.mse_loss(p, t) for p, t in zip(pred_features, target_features))

    def style_loss(self, pred, target):
        pred = self.replicate_grayscale(pred)
        target = self.replicate_grayscale(target)
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        
        def gram_matrix(x):
            b, c, h, w = x.size()
            features = x.view(b, c, h * w)
            return torch.bmm(features, features.transpose(1, 2)) / (c * h * w)
        
        return sum(F.mse_loss(gram_matrix(p), gram_matrix(t)) for p, t in zip(pred_features, target_features))

    def adversarial_loss(self, pred, target, discriminator):
        real_loss = -torch.log(1 - discriminator(target, pred) + 1e-10).mean()
        fake_loss = -torch.log(discriminator(pred, target) + 1e-10).mean()
        return real_loss + fake_loss
