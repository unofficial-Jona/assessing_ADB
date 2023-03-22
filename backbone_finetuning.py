#%% 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from torchvision.io import read_image
import numpy as np
import json
import pickle
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

from pdb import set_trace

# %% [markdown]
# # model setup
# loading model and replacing head for fine-tuning 

#%% 

model = swin_v2_b(weights= Swin_V2_B_Weights.DEFAULT)
model.head = nn.Linear(model.head.in_features, 4)

# %% [markdown]
# # prepare dataset
# the loader is supposed to iterate through the train set only
# load annotations in init function
# create list with sublists: [vid_name, frame, anno]
# load image in __getitem__(self, idx) where idx indicates the sublist to access
# ensure that if no .jpeg file is available, something (zeros) is returned


# %%
class METEOR_Frames(Dataset):
    def __init__(self, info_loc = '../../pvc-meteor/features/' , data_loc = '../../pvc-meteor/frames/' , anno_loc = '../../pvc-meteor/features/'):
        
        info_file = json.load(open(os.path.join(info_loc, 'METEOR_info.json'), 'r'))
        anno_file = pickle.load(open(os.path.join(anno_loc, 'new_anno_2.pkl'), 'rb'))

        vid_names = info_file['METEOR']['train_session_set']
        self.inputs = list()

        # iterate through video names
        for vid in vid_names:
            if vid not in anno_file.keys():
                continue

            # iterate through index
            for i in range(anno_file[vid]['feature_length']):
                # store vid_name, frame_name and anno in self.inputs
                frame_name = f'frame_{i:06d}.jpg'
                anno = anno_file[vid]['anno'][i,:]

                # only intersted in subset with frequent categories --> use indices as used in METEORDataLayer
                anno = anno[[0,2,4,6]]

                self.inputs.append([vid, frame_name, anno])

        self.data_loc = data_loc

    def __len__(self):
        """
        Returns the number of data items in the dataset.
        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieves a data item and its label at the specified index.

        Args:
            idx (int): Index of the data item to retrieve

        Returns:
            tuple: (data item, label)
        """
        vid, frame_name, anno = self.inputs[idx]
        # need to cut .MP4 ending from video name to get proper file name
        img_path = os.path.join(self.data_loc, vid[:-4], frame_name)
        # try to read image, if not possible set hand over zeros and set annotations to zero.
        try:
            image = read_image(img_path)
        except:
            image = torch.zeros((416, 234))
            anno = np.zeros_like(anno)

        return image, anno


# %%
dataset = METEOR_Frames()
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)


# Set up the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

# Training parameters
num_epochs = 5
log_interval = 1000

# Calculate total number of batches
total_batches = len(dataloader)
print(f"Total batches per epoch: {total_batches}")


# %% 
# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    epoch_start_time = time.time()

    for batch_idx, (images, annotations) in enumerate(dataloader):
        # Move data to device
        images = images.to(device)
        annotations = annotations.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, annotations)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Logging
        running_loss += loss.item()
        if batch_idx % log_interval == log_interval - 1:
            avg_loss = running_loss / log_interval
            avg_time_per_batch = (time.time() - epoch_start_time) / log_interval
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}], Loss: {avg_loss:.4f}, Avg Time per Batch: {avg_time_per_batch:.4f} seconds")
            running_loss = 0.0

    epoch_time = time.time() - epoch_start_time
    print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Time taken: {epoch_time:.2f} seconds")


# %%
# Save the model
checkpoint_path = "pretrained_backbone.pth"
torch.save(model.state_dict(), checkpoint_path)
print(f"Model saved at {checkpoint_path}")
print(f"Model saved at {checkpoint_path}")