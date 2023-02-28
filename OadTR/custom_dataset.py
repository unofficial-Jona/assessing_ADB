import os.path as osp
import pickle
import torch
import torch.utils.data as data
import numpy as np
import json
from tqdm import tqdm


import warnings

from pdb import set_trace

def get_weights(target, session_names, use_idx):
    total_frames = 0
    frames_per_cat = np.zeros(len(use_idx))
    # makes sure that only videos in training data are used to calculate weights
    for name in session_names:
        total_frames += target[name]['feature_length']
        _anno = target[name]['anno'][:, use_idx].sum(axis=0)
        frames_per_cat += _anno

    frames_not_cat = np.array([total_frames for i in use_idx]) - frames_per_cat
    weights = frames_not_cat / frames_per_cat
    # weights /= weights.min()
    # print(f'using the following weights: {weights}')
    # warnings.warn('weights are scaled by 3')
    return torch.tensor(weights)


class METEORDataLayer(data.Dataset):
    def __init__(self, args, phase='train') -> None:
        
        
        self.data_root = '../../../pvc-meteor/features'
        self.pickle_root = '../../../pvc-meteor/features'
        self.sessions = getattr(args,phase + '_session_set') # used to get video names from json file --> no need to split into train and test
        self.enc_steps = args.enc_layers
        self.dec_steps = args.query_num
        self.inputs = list()
        
        # print(f'{phase}: {len(self.sessions)} videos')
        
        # load file and split into annotations and features
        
        '''
        with open(osp.join(self.pickle_root, f'METEOR.pickle'), 'rb') as f:
            file_size = osp.getsize(osp.join(self.pickle_root, f'METEOR.pickle'))
            with tqdm.wrapattr(f, "read", total=file_size, desc="File content") as file:
                load_obj = pickle.load(file)

        
        '''
        load_obj = pickle.load(open(osp.join(self.pickle_root, args.pickle_file_name), 'rb'))
        
        target_all = load_obj['annotations']
        self.feature_all = load_obj['features']
        feature_dict = load_obj['meta']
        feature_dict['fps'] = int(30 / feature_dict['fps'])
        args.feature = feature_dict


        assert not (args.use_frequent and args.use_infrequent), "can't select frequent and infrequent categories at the same time"
        
        # define index that should be used
        if args.use_frequent:
            self.use_idx = [0,2,4,6]
        
        elif args.use_infrequent:
            warnings.warn("args.all_class_names (defined in main) may needs to be updated.")
            self.use_idx = [1,3,5]
        
        else:
            warnings.warn("args.all_class_names (defined in main) may needs to be updated.")
            self.use_idx = [0,1,2,3,4,5,6]
            
        # set up weights for loss function
        self.weights = get_weights(target_all, self.sessions, self.use_idx)
        
        
        # load and prepare annotations
        for session in self.sessions:
            target = target_all[session]['anno'][:, self.use_idx]
            seed = np.random.randint(self.enc_steps) if phase == 'train' else 0
            for start, end in zip(
                    range(seed, target.shape[0], 1),  # self.enc_steps
                    range(seed + self.enc_steps, target.shape[0]-self.dec_steps, 1)):
                enc_target = target[start:end]
                # dec_target = self.get_dec_target(target[start:end + self.dec_steps])
                dec_target = target[end:end + self.dec_steps]
                distance_target, class_h_target = self.get_distance_target(target[start:end])
                if class_h_target.argmax() != 21:
                    self.inputs.append([
                        session, start, end, enc_target, distance_target, class_h_target, dec_target
                    ])

        

            
    def get_dec_target(self, target_vector):
        target_matrix = np.zeros((self.enc_steps, self.dec_steps, target_vector.shape[-1]))
        for i in range(self.enc_steps):
            for j in range(self.dec_steps):
                # 0 -> [1, 2, 3]
                # target_matrix[i,j] = target_vector[i+j+1,:]
                # 0 -> [0, 1, 2]
                target_matrix[i, j] = target_vector[i + j, :]
        return target_matrix

    def get_distance_target(self, target_vector):
        target_matrix = np.zeros(self.enc_steps - 1)
        # here only one class is encoded --> needs to be changed
        target_argmax = target_vector[self.enc_steps-1].argmax()
        for i in range(self.enc_steps - 1):
            if target_vector[i].argmax() == target_argmax:
                target_matrix[i] = 1.
        return target_matrix, target_vector[self.enc_steps-1]

    def __getitem__(self, index):
        '''self.inputs.append([
                    session, start, end, enc_target, distance_target, class_h_target
                ])'''
        session, start, end, enc_target, distance_target, class_h_target, dec_target = self.inputs[index]
        
        camera_inputs = self.feature_all[session]['rgb'][start:end]
        camera_inputs = torch.tensor(camera_inputs)
        
        motion_inputs = self.feature_all[session]['flow'][start:end]
        motion_inputs = torch.tensor(motion_inputs)
        
        enc_target = torch.tensor(enc_target)
        distance_target = torch.tensor(distance_target)
        class_h_target = torch.tensor(class_h_target)
        dec_target = torch.tensor(dec_target)

        return camera_inputs, motion_inputs, enc_target, distance_target, class_h_target, dec_target

    def __len__(self):
        return len(self.inputs)
    
    
class ARGS():
    def __init__(self):
        self.phase='train'
        self.num_layers= 2
        self.enc_layers = 2
        self.numclass = 4
        self.query_num = 8
        self.train_session_set = json.load(open('../../../pvc-meteor/features/METEOR_info.json', 'r'))['METEOR']['train_session_set']
        self.use_frequent = True
        self.use_infrequent = False
    
if __name__ == '__main__':

    args = ARGS()
    
    data_layer = METEORDataLayer(args)