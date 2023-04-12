import os.path as osp
import pickle
import torch
import torch.utils.data as data
import numpy as np
import json
from tqdm import tqdm
import pandas as pd


import warnings

from pdb import set_trace

def get_weights(target, session_names, use_idx):
    total_frames = 0
    frames_per_cat = np.zeros(len(use_idx))
    # makes sure that only videos in training data are used to calculate weights
    for name in session_names:
        if name not in target.keys():
            continue
        total_frames += target[name]['feature_length']
        _anno = target[name]['anno'][:, use_idx].sum(axis=0)
        frames_per_cat += _anno

    frames_not_cat = np.array([total_frames for i in use_idx]) - frames_per_cat
    weights = frames_not_cat / frames_per_cat
    # weights /= weights.min()
    # print(f'using the following weights: {weights}')
    # warnings.warn('weights are scaled by 3')
    return torch.tensor(weights)

def get_weights_recent_frame(inputs):
    ### WEIGHT BASED ON MOST RECENT FRAME
    total_frames = 0
    class_count = np.zeros((5))
    for input in inputs:
        _, _, _, target, _, _, _ = input

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
    total_frames = 0
    class_count = np.zeros((5))
    for input in inputs:
        _, _, _, target, _, _, _ = input
        target_vector = target.sum(axis=0)
        # set_trace()
        assert target_vector.shape == class_count.shape, 'target shape is off'

        class_count += target_vector
        total_frames += target.shape[0]
    
    frames_not_cat = total_frames - class_count

    assert frames_not_cat.min() >= 0, 'too few frames per category'

    weight_vector = frames_not_cat / class_count
    return torch.tensor(weight_vector)


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
        

        load_obj = pickle.load(open(osp.join(self.pickle_root, args.pickle_file_name), 'rb'))
        
        # target_all = pickle.load(open(osp.join(self.pickle_root, 'new_anno.pkl'), 'rb'))
        target_all = load_obj['annotations']
        
        warnings.warn('modified to incorporate background class')
        
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
        
        
        # set_trace()
        
        # load and prepare annotations
        for session in self.sessions:
            
            # modification to accomodate annotation_file with fewer files/annotations 
            if session not in target_all.keys():
                continue
            
            target = target_all[session]['anno'][:, self.use_idx]
            #### MODIFICATIONS BACKGROUND CLASS  ####
            new_target = np.zeros((target.shape[0], target.shape[1] + 1))
            new_target[:,1:] = target
            
            assert np.sum(new_target[:,0]) == 0, 'first row is not 0'
            
            background_vector = np.sum(target, axis=1).clip(0,1).astype(bool)
            new_target [:, 0] = ~background_vector
            
            target = new_target
            
            seed = np.random.randint(self.enc_steps) if phase == 'train' else 0
            for start, end in zip(
                    range(seed, target.shape[0], 1),  # self.enc_steps
                    range(seed + self.enc_steps, target.shape[0]-self.dec_steps, 1)):
                enc_target = target[start:end]
                # dec_target = self.get_dec_target(target[start:end + self.dec_steps])
                dec_target = target[end:end + self.dec_steps]
                distance_target, class_h_target = self.get_distance_target(target[start:end])
                self.inputs.append([session, start, end, enc_target, distance_target, class_h_target, dec_target])


            
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
        # target_argmax = target_vector[self.enc_steps-1]
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
        
        for return_tensor in [camera_inputs, motion_inputs, enc_target, distance_target, class_h_target, dec_target]:
            if return_tensor.dtype == torch.float16:
                return_tensor = return_tensor.to(torch.float32)
        
        return camera_inputs, motion_inputs, enc_target, distance_target, class_h_target, dec_target

    def __len__(self):
        return len(self.inputs)

class METEOR_3D(data.Dataset):
    # class loading features extracted with 3D resnet --> __getitem__ should return same shape as 'original' METEORDataLayer.
    def __init__(self, args, phase='train', weights=False) -> None:
        self.sessions = getattr(args,phase + '_session_set')
        self.enc_steps = args.enc_layers
        self.dec_steps = args.query_num
        self.inputs = list()
        
        load_obj = pickle.load(open('/workspace/pvc-meteor/features/colar/extraction_output_colar.pkl', 'rb'))
        
        self.feature_all = load_obj['features']
        target_all = load_obj['annotations']
        feature_dict = load_obj['meta']
        feature_dict['fps'] = int(30 / feature_dict['fps'])
        args.feature = feature_dict
        
        self.use_idx = [0,2,4,6]
        
        for session in self.sessions:
            
            target = target_all[session]['anno'][:, self.use_idx]
            new_target = np.zeros((target.shape[0], target.shape[1] + 1))
            new_target[:,1:] = target
            
            assert np.sum(new_target[:,0]) == 0, 'first row is not 0'
            
            background_vector = np.sum(target, axis=1).clip(0,1).astype(bool)
            new_target [:, 0] = ~background_vector
            
            target = new_target
            seed = np.random.randint(self.enc_steps) if phase == 'train' else 0
            for start, end in zip(
                    range(seed, target.shape[0], 1),  # self.enc_steps
                    range(seed + self.enc_steps, target.shape[0]-self.dec_steps, 1)):
                enc_target = target[start:end]
                # dec_target = self.get_dec_target(target[start:end + self.dec_steps])
                dec_target = target[end:end + self.dec_steps]
                distance_target, class_h_target = self.get_distance_target(target[start:end])
                self.inputs.append([session, start, end, enc_target, distance_target, class_h_target, dec_target])
                
        if weights == False:
            self.weights = None
        elif weights.lower() == 'recent':
            self.weights = get_weights_recent_frame(self.inputs)
        elif weights.lower() == 'all':
            self.weights = get_weights_entire_inputs(self.inputs)
        else:
            assert False, 'invalid argument for weights'

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
        # target_argmax = target_vector[self.enc_steps-1]
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
        
        camera_inputs = self.feature_all[session]['features'][start:end]
        camera_inputs = torch.tensor(camera_inputs)
        
        # motion_inputs = self.feature_all[session]['flow'][start:end]
        motion_inputs = torch.empty_like(camera_inputs)
        
        enc_target = torch.tensor(enc_target)
        distance_target = torch.tensor(distance_target)
        class_h_target = torch.tensor(class_h_target)
        dec_target = torch.tensor(dec_target)
        
        for return_tensor in [camera_inputs, motion_inputs, enc_target, distance_target, class_h_target, dec_target]:
            if return_tensor.dtype == torch.float16:
                return_tensor = return_tensor.to(torch.float32)
        
        return camera_inputs, motion_inputs, enc_target, distance_target, class_h_target, dec_target
    def __len__(self):
        return len(self.inputs)
    
    
class PURE_METEORDataLayer(data.Dataset):
    def __init__(self, args, phase='test') -> None:
        
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
                # set_trace()

                # dec_target = self.get_dec_target(target[start:end + self.dec_steps])
                dec_target = target[end:end + self.dec_steps]
                distance_target, class_h_target = self.get_distance_target(target[start:end])
                
                if np.sum(class_h_target) != 1:
                    continue
                else:
                    self.inputs.append([
                        session, start, end, enc_target, distance_target, class_h_target, dec_target
                    ])

        print(f'testing with {len(self.inputs)} pure examples')

            
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
        # target_argmax = target_vector[self.enc_steps-1]
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
        self.phase='test'
        self.num_layers= 2
        self.enc_layers = 2
        self.numclass = 4
        self.query_num = 8
        self.train_session_set = json.load(open('../../../pvc-meteor/features/METEOR_info.json', 'r'))['METEOR']['train_session_set']
        self.use_frequent = True
        self.use_infrequent = False
    
if __name__ == '__main__':

    args = ARGS()
    args.pickle_file_name = 'extraction_output_22-02-2023-16-18.pkl'
    data_layer = METEORDataLayer(args)
    # sampler_val = torch.utils.data.SequentialSampler(data_layer)
    loader = data.DataLoader(data_layer, 1)
    
    
    print('what a wonderful world')