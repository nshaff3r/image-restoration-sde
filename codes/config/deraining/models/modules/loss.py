import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import sys


class MatchingLoss(nn.Module):
    def __init__(self, vgg_model=None):
        super().__init__()
        if vgg_model is None:
            vgg_model = models.vgg16(pretrained=True).features
        self.vgg = vgg_model
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.lambda_rec = 1
        self.lambda_per = 0.2
        self.lambda_sty = 250
        self.lambda_pyr1 = self.lambda_pyr2 = self.lambda_pyr3 = 1
        self.lambda_adv = 0.2

    def forward(self, pred, target, discriminator=None):
        # Reconstruction Loss
        l_rec = F.l1_loss(pred, target)
        
        # Perceptual Loss
        l_per = self.perceptual_loss(pred, target)
        
        # Style Loss
        l_sty = self.style_loss(pred, target)
        
        # Pyramid Loss
        l_pyr = self.lambda_rec * l_rec + self.lambda_per * l_per + self.lambda_sty * l_sty
        
        # Adversarial Loss
        l_adv = torch.tensor(0.0, device=pred.device)
        if discriminator is not None:
            l_adv = self.adversarial_loss(pred, target, discriminator)
        
        # Total Loss
        total_loss = (self.lambda_pyr1 + self.lambda_pyr2 + self.lambda_pyr3) * l_pyr + self.lambda_adv * l_adv
        
        return total_loss.mean()  # Take the mean of the total loss

    def perceptual_loss(self, pred, target):
        losses = []
        for i, layer in enumerate(self.vgg):
            pred = layer(pred)
            target = layer(target)
            if isinstance(layer, nn.MaxPool2d):
                losses.append(F.l1_loss(pred, target))
        return torch.mean(torch.stack(losses))

    def style_loss(self, pred, target):
        def gram_matrix(x):
            b, c, h, w = x.size()
            features = x.view(b, c, h * w)
            gram = torch.bmm(features, features.transpose(1, 2))
            return gram.div(c * h * w)

        losses = []
        for i, layer in enumerate(self.vgg):
            pred = layer(pred)
            target = layer(target)
            if isinstance(layer, nn.Conv2d):
                losses.append(F.l1_loss(gram_matrix(pred), gram_matrix(target)))
        return torch.mean(torch.stack(losses))

    def adversarial_loss(self, pred, target, discriminator):
        real_loss = -torch.log(1 - discriminator(target, pred) + 1e-10).mean()
        fake_loss = -torch.log(discriminator(pred, target) + 1e-10).mean()
        return real_loss + fake_loss
