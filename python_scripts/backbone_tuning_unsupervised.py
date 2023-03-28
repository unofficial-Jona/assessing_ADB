import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale, GaussianBlur
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import gc
from pdb import set_trace
import kornia.augmentation as K
from tqdm import tqdm
from pdb import set_trace

from dataset import METEOR_Frames

BATCH_SIZE = 4


# class to compute contrastive loss between two samples
# TODO: check if temperature = 0.5 is a good value
class NTXentLoss(torch.nn.Module):
    def __init__(self, batch_size=BATCH_SIZE, temperature=0.1, n_views=2):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.n_views = n_views

    def forward(self, features):
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(features.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

        logits = logits / self.temperature

        loss = F.cross_entropy(logits, labels)
        return loss

    
# image augmentations
def simclr_augmentations(size=(416, 234)):
    return nn.Sequential(
        K.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
        K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        K.RandomGrayscale(p=0.2),
        K.RandomGaussianBlur(kernel_size=int(0.1 * size[0]), sigma=(0.1, 2.0)),
    )

dataset = METEOR_Frames()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


# Load the base model
model = swin_v2_b(weights = Swin_V2_B_Weights.DEFAULT)

# set head to nn.Identity to obtain latent features instead of 1000 classes
model.head = nn.Identity()



# Define the optimizer and loss function
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = NTXentLoss()
scaler = GradScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

augmentations = simclr_augmentations().to(device)

epochs = 2
save_interval = 1000
log_interval = 2


log_file = 'unsupervised_tuning_log.txt'


print("start training")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    epoch_start_time = time.time()
    print(f'Epoch [{epoch + 1}/ {num_epochs}]')
    pbar = tqdm(dataloader, desc=f"Loss: N/A")
    for idx, (images, _) in enumerate(pbar):
        images = images.to(device)
        img1 = augmentations(images)
        img2 = augmentations(images)

        # Forward pass
        optimizer.zero_grad()
        
        with autocast():
            z1 = model(img1)
            z2 = model(img2)

            loss = criterion(torch.cat([z1, z2], dim=0))

        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        pbar.set_description(f"Loss: {loss.item():.4f}")

        if idx % log_interval == 0:
            avg_loss = total_loss / log_interval
            with open(log_file, "a") as f:
                f.write(f"Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {loss:.4f}, Avg Loss: {avg_loss:.4f}\n")
            total_loss = 0
        
        if idx % save_interval == 0:
            torch.save(model.state_dict(), f"unsupervised_backbone.pth")
    
    epoch_time = time.time() - epoch_start_time    
    print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Time taken: {epoch_time:.2f} seconds")
