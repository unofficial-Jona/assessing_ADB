# this script is intended to replace the dataset.py in the OadTR repository
# For legacy reasons the main class will be called TRNTHUMOSDataLayer, despite having nothing to do with THUMOS data


import os.path as osp
import pickle
import torch
import torch.utils.data as data
import numpy as np
from ipdb import set_trace


class TRNTHUMOSDataLayer(data.Dataset):
    def __init__(self, args, phase='train'):
        args.data_root = 'data'
        self.pickle_root = 'data'
        self.data_root = args.data_root  # data/THUMOS
        self.sessions = getattr(args, phase + '_session_set')  # video name
        self.enc_steps = args.enc_layers
        self.numclass = args.numclass
        self.dec_steps = args.query_num
        self.training = phase == 'train'
        self.feature_pretrain = args.feature  # 'Anet2016_feature'   # IncepV3_feature  Anet2016_feature
        self.inputs = []
        
        self.subnet = 'val' if self.training else 'test'

        # preparing targets
        target_all = pickle.load(open(osp.join(self.pickle_root, 'thumos_' + self.subnet + '_anno.pickle'), 'rb'))
        for session in self.sessions:  # 改
            # target = np.load(osp.join(self.data_root, 'target', session+'.npy'))  # thumos_val_anno.pickle
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


        # preparing inputs 
        assert osp.exists(osp.join(self.pickle_root, 'thumos_all_feature_{}_V3.pickle'.format(self.subnet))), f"can't find file: thumos_all_feature_{self.subnet}_V3.pickle in directory {self.pickle_root}"
        self.feature_All = pickle.load(open(osp.join(
            self.pickle_root, 'thumos_all_feature_{}_V3.pickle'.format(self.subnet)), 'rb'))
    


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
        camera_inputs = self.feature_All[session]['rgb'][start:end]
        camera_inputs = torch.tensor(camera_inputs)
        motion_inputs = self.feature_All[session]['flow'][start:end]
        
        motion_inputs = torch.tensor(motion_inputs)
        enc_target = torch.tensor(enc_target)
        distance_target = torch.tensor(distance_target)
        class_h_target = torch.tensor(class_h_target)
        dec_target = torch.tensor(dec_target)
        return camera_inputs, motion_inputs, enc_target, distance_target, class_h_target, dec_target

    def __len__(self):
        return len(self.inputs)
