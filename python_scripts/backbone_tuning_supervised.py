#%% 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from torch.optim import Adam
from torchvision.io import read_image
import numpy as np
import json
import pickle
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import time
import gc
from pdb import set_trace
from dataset import METEOR_Frames
from tqdm import tqdm

from torch.cuda import amp

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMallocAsync'


# %% [markdown]
# # model setup
# loading model and replacing head for fine-tuning 

#%% 
def train_one_epoch_supervised():
    BATCH_SIZE = 64
    checkpoint_path = "supervised_backbone.pth"
    log_file = 'supervised_tuning_log.txt'

    # Training parameters#
    log_interval = 5
    save_interval = 1000

    
    model = swin_v2_b(weights= Swin_V2_B_Weights.DEFAULT)
    model.head = nn.Linear(model.head.in_features, 4)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            old_log = f.read()
            epoch = int(old_log.split('\n')[-2].split(',')[0].split(':')[-1])
    else:
        epoch = 0

    # disabled cause of downgrade from pytorch 2.0.0 to 1.13
    # model = torch.compile(model)

    # %% [markdown]
    # # prepare dataset
    # the loader is supposed to iterate through the train set only
    # load annotations in init function
    # create list with sublists: [vid_name, frame, anno]
    # load image in __getitem__(self, idx) where idx indicates the sublist to access
    # ensure that if no .jpeg file is available, something (zeros) is returned

    # %%
    dataset = METEOR_Frames()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)


    # %% 
    # Set up the model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight = dataset.get_weights()).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)


    # Create a GradScaler for mixed precision training
    scaler = amp.GradScaler()

    # %% 
    # Training loop
    print(f'start training Epoch {epoch + 1}')
    model.train()
    running_loss = 0.0
    epoch_start_time = time.time()
    total_loss = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), smoothing=0)
    for batch_idx, (images, annotations) in progress_bar:
        # Move data to device
        images = images.to(device)
        annotations = annotations.to(device)

        # Forward pass
        optimizer.zero_grad()
        
        # Use amp.autocast for mixed precision training
        with amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, annotations)

        # Use the scaler to scale the loss and perform the backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        # Clear GPU cache
        # torch.cuda.empty_cache()

        # Delete tensors and call garbage collection
        # del images, annotations, outputs
        # gc.collect()

        # Logging
        running_loss += loss.item()
        total_loss += loss.item()
        
        pbar_loss = total_loss/(batch_idx + 1)
        
        progress_bar.set_description(f"Loss: {pbar_loss:.4f}")
        
        
        if batch_idx % log_interval == 0:
            avg_loss = running_loss / log_interval
            with open(log_file, "a") as f:
                f.write(f"Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {loss:.4f}, Avg Loss: {avg_loss:.4f}\n")
            running_loss = 0.0
        if batch_idx % save_interval == 0:
            torch.save(model.state_dict(), checkpoint_path)

        # Update the progress bar description with the average loss

        
    epoch_time = time.time() - epoch_start_time

    print(f"Epoch {epoch + 1} completed. Time taken: {epoch_time:.2f} seconds")

if __name__ == '__main__':
    train_one_epoch_supervised()