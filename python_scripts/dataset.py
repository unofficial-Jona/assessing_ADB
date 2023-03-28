from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import json
import pickle
import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from pdb import set_trace
import torch
from torchvision.transforms import ToPILImage
# from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale, GaussianBlur


class METEOR_Frames(Dataset):
    def __init__(self, session='train', info_loc = '../../../pvc-meteor/features/' , data_loc = '../../../pvc-meteor/frames/' , anno_loc = '../../../pvc-meteor/features/', transform=None):
        
        info_file = json.load(open(os.path.join(info_loc, 'METEOR_info.json'), 'r'))
        anno_file = pickle.load(open(os.path.join(anno_loc, 'new_anno_2.pkl'), 'rb'))

        vid_names = info_file['METEOR'][f'{session}_session_set']
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
        self.transform = transform
        self.data_loc = data_loc
    
    def get_weights(self):
        frame_count = np.zeros(4)
        total_frames = 0
        for (_,_, anno) in self.inputs:
            frame_count += anno
            total_frames += 1
            
        not_frame_count = np.full_like(frame_count, total_frames) - frame_count
        weights = torch.tensor(not_frame_count/frame_count)
        return weights / weights.min()
    
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

        # normalize image to range [0,1]
        image = image / 255
        
        if self.transform:
            image = self.transform(image)
        
        return image, anno
    
    
if __name__ == '__main__':
    dataset = METEOR_Frames()
    dataloader = DataLoader(dataset, batch_size=2)
    print(dataset.get_weights())
    img, anno = next(iter(dataloader))
    to_pil = ToPILImage()
    
    plt.imshow(to_pil(img[0]))
    print(f'annotation: {anno[0]}')