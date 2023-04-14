import os.path as osp
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from pdb import set_trace

def get_weights_recent_frame(inputs):
    ### WEIGHT BASED ON MOST RECENT FRAME
    total_frames = 0
    class_count = np.zeros((5))
    for input in inputs:
        _, _, _, target = input

        target_vector = target[-1, :]

        assert target_vector.shape == class_count.shape, 'target shape is off'

        class_count += target_vector
        total_frames += 1

    # get frames not cat
    frames_not_cat = total_frames - class_count

    assert frames_not_cat.min() >= 0, 'too few frames per category'

    weight_vector = frames_not_cat / class_count
    return torch.tensor(weight_vector)

def get_weights_entire_inputs(inputs):
    ### WEIGHT BASED ON ENTIRE INPUTS
    # iterate through inputs and sum enc_target along axis 0
    total_frames = 0
    class_count = np.zeros((5))
    for input in inputs:
        _, _, _, target = input
        target_vector = target.sum(axis=0)
        # set_trace()
        assert target_vector.shape == class_count.shape, 'target shape is off'

        class_count += target_vector
        total_frames += target.shape[0]
    
    frames_not_cat = total_frames - class_count

    assert frames_not_cat.min() >= 0, 'too few frames per category'

    weight_vector = frames_not_cat / class_count
    return torch.tensor(weight_vector)

class MeteorDataset(Dataset):
    def __init__(self, args, flag='train', use_frequent=True, weights=False):
        assert flag in ['train', 'test']
        self.pickle_root = args.data_root
        self.sessions = getattr(args, flag + '_session_set')
        self.enc_steps = args.enc_layers
        self.training = flag == 'train'
        
        self.inputs = []

        self.subnet = 'val' if self.training else 'test'

        load_obj = pickle.load(open(self.pickle_root, 'rb'))
        
        target_all = load_obj['annotations']
        self.feature_all = load_obj['features']
        
        # set_trace()
        
        for session in self.sessions:
            target = target_all[session]['anno']
            if use_frequent:
                target = target[:,[0,2,4,6]]

            assert target.shape[0] == target_all[session]['feature_length'], 'feature shape is off'
            
            # obtain background class
            background_vector = target.sum(axis=1).clip(0,1).astype(bool)
            new_target = np.zeros((target.shape[0], target.shape[1] + 1))
            new_target[:,0] = ~background_vector
            new_target[:,1:] = target
            
            target = new_target
                                             
            seed = np.random.randint(self.enc_steps) if self.training else 0
            for start, end in zip(
                    range(seed, target.shape[0], 1),
                    range(seed + self.enc_steps, target.shape[0], 1)):  # target.shape[0]
                enc_target = target[start:end]
                class_h_target = enc_target[self.enc_steps - 1]
                self.inputs.append([session, start, end, enc_target])
        if weights == False:
            self.weights = None
        elif weights.lower() == 'recent':
            self.weights = get_weights_recent_frame(self.inputs)
        elif weights.lower() == 'all':
            self.weights = get_weights_entire_inputs(self.inputs)
        else:
            assert False, 'invalid argument for weights'
            

    def __getitem__(self, index):
        session, start, end, enc_target = self.inputs[index]
        camera_inputs = self.feature_all[session]['features'][start:end]
        camera_inputs = torch.tensor(camera_inputs)

        enc_target = torch.tensor(enc_target)

        return camera_inputs, enc_target

    def __len__(self):
        return len(self.inputs)


if __name__ == '__main__':
    from misc.config import parse_args
    import json

    args = parse_args()
    with open(args.dataset_file, 'r') as f:
        data_info = json.load(f)['METEOR']

    args.train_session_set = data_info['train_session_set']
    args.test_session_set = data_info['test_session_set']
    dataset = MeteorDataset(args)

    sampler_val = torch.utils.data.SequentialSampler(dataset)
    data_loader_val = DataLoader(dataset, args.batch, sampler=sampler_val,
                                 drop_last=False, pin_memory=True, num_workers=8)
    for feature, target in data_loader_val:
        print(feature.shape, target.shape)
        break
