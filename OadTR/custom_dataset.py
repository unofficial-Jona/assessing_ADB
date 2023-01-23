import os.path as osp
import pickle
import torch
import torch.utils.data as data
import numpy as np

class METEORDataLayer(data.Dataset):
    def __init__(self, args, phase='train') -> None:
        self.data_root = '../../pvc-meteor/features'
        self.pickle_root = '../../pvc-meteor/features'
        self.sessions = getattr(args,phase + '_session_set') # used to get video names from json file
        self.enc_steps = args.enc_layers
        self.encoder_steps = args.enc_layers
        self.numclass = args.numclass
        self.dec_steps = args.query_num
        self.training = phase == 'train'
        self.inputs = list()

        # load annotation file based on phase
        self.subnet = 'train' if self.training else 'test'
        target_all = pickle.load(open(osp.join(self.pickle_root, f'target_METEOR_{self.subnet}.pickle'), 'rb'))
        
        

        for session in self.sessions:
            target = target_all[session]['anno']
            seed = np.random.randint(self.enc_steps) if self.training else 0
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

        self.feature_all = pickle.load(open(osp.join(self.pickle_root, f'feature_METEOR_{self.subnet}.pickle'), 'rb'))
        
                    
            
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