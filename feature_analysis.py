import argparse
from OadTR.custom_config import get_args_parser
from OadTR.custom_dataset import METEOR_3D, METEORDataLayer
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle


def analyse_nr_features():
    parser = argparse.ArgumentParser('OadTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    with open(args.dataset_file, 'r') as f:
        data_info = json.load(f)['METEOR']
    args.train_session_set = data_info['train_session_set']
    args.test_session_set = data_info['test_session_set']

    result_dict = {}

    for pkl_file in ['features_att_15_new.pkl','features_att_30_new.pkl','features_conv_15_new.pkl','features_i3d.pkl','features_TSN.pkl']:
        print(pkl_file)
        args.pickle_file_name = pkl_file

        pkl_dict = {}
        for phase in ['train', 'test']:
            
            if pkl_file == 'features_i3d.pkl':
                dataset = METEOR_3D(phase=phase, args=args)
            else:
                dataset = METEORDataLayer(phase=phase, args=args)
            instances = len(dataset.inputs)
            print(f'\t{phase}: {instances}')
            pkl_dict.update({phase: instances})
        result_dict.update(pkl_dict)
    return result_dict


if __name__ == '__main__':
    out_dict = analyse_nr_features()
    out_df = pd.DataFrame(out_dict)
    ax = out_df.plot.bar()
        
    fig = ax.get_figure()
    fig.savefig('/workspace/pvc-meteor/features/feature_instances.png')