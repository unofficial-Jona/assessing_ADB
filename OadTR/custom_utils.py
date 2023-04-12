# %%
import json
import ast
import torch
from torch.utils.data import DataLoader
import itertools
import os
from tqdm import tqdm
from pdb import set_trace
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, f1_score, precision_score, recall_score, label_ranking_average_precision_score, coverage_error

from custom_dataset import METEORDataLayer, PURE_METEORDataLayer
import transformer_models



# %%

def generate_dict(location):
    try:
        with open(location, 'r') as f:
            doc = f.read()
    except IsADirectoryError:
        with open(os.path.join(location , 'log_dist.txt'), 'r') as f:
            doc = f.read()

    arg_str = doc.split('\nnumber of params')[0]
    arg_str = arg_str.split('\n')
    arg_str = [i.split(':') for i in arg_str[1:]]

    new_dic = {}

    for i in arg_str:
        if len(i) != 2:
            continue
        new_dic[i[0]] = i[1]
    new_dic

    for k, v in new_dic.items():
        
        if v.lower() == 'true':
            new_dic[k] = True
        
        elif v.lower() == 'false':
            new_dic[k] = False
        
        elif v.lower() == 'none':
            new_dic[k] = None
        
        elif 'torch' and 'tensor' in v:
            my_string = v.replace('tensor(', '').replace(', dtype=torch.float64)', '')
            my_object = ast.literal_eval(my_string)
            new_dic[k] = torch.tensor(my_object)
        
        elif '.' in v and (not '/' in v and not '_' in v):
            new_dic[k] = float(v)
        
        elif '[' and ']' in v and not 'torch' in v:
            new_dic[k] = ast.literal_eval(v)
        
        elif v.isnumeric():
            new_dic[k] = int(v)


    return new_dic

class ModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

def get_args(exp_dir):
    dic = generate_dict(exp_dir)
    return ModelConfig(**dic)
        
        
def get_multilabel_conf_mat(y_true, y_pred, label_names, save_loc='', visualize=False):
    """ convenience wrapper for skelarn.metrics.multilabel_confusion_matrix. Can visualize and 

    Args:
        y_true (np.array): Ground Truth
        y_pred (np.array): Predictions
        label_names (list): len(label_names) must match nr. classes
        save (str, optional): If set saves the plot at the given location. Defaults to '', meaning no save.
        visualize (bool, optional): Whether to visualize the output. Defaults to False.

    Returns:
        np.array: confusion matrices as returned by sklearn.metrics.multilabel_confusion_matrix
    """
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(confusion_matrices.shape[0], 1, figsize=(15,15))
                        
    for i, (mat, lab) in enumerate(zip(confusion_matrices, label_names)):
        disp = ConfusionMatrixDisplay(mat, display_labels=[lab + '-','+'+lab])
        disp.plot(ax=ax[i], values_format='.0f', cmap='Blues', xticks_rotation='horizontal')
        disp.ax_.set_title(lab)
        disp.im_.colorbar.remove()
    plt.tight_layout()
    if visualize:
        plt.show()
    if save_loc:
        plt.savefig(os.path.join(save_loc, 'confusion_matrices.png')) #confusion matrices
    return confusion_matrices

def eval_experiment(exp_dir, args, save_results=True, save_fig=True, checkpoint='DEFAULT'):
    """evaluate an experiment generated run on the OadTR model. 
    Metrics included in this evaluation are: precision, recall, f1, roc_auc, coverage_error, label_ranking_avg_precision, confusion_matrices.

    Args:
        exp_dir (str): directory of the experiment
        save_results (bool, optional): whether to save the experiment results. Defaults to True.
        save_fig (bool, optional): whether to save the confusion matrices. Defaults to True.
        checkpoint (str or int, optional): which checkpoint to load. Default loads latest checkpoint, otherwise loads specified number. Defaults to 'DEFAULT'.

    Returns:
        dict: Dictionary with all relevant metrics.
    """
    # evaluate a single experiment with the provided data.
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # prepare dataloader:
    if 'pickle_file_name' not in args.__dict__.keys():
        if args.dim_feature == 2048 and '30_FPS' not in exp_dir: # attention based backbone, not 30 FPS
            args.pickle_file_name = 'extraction_output_11-02-2023-18-33.pkl'
        elif args.dim_feature == 2048 and '30_FPS' in exp_dir: # attention based backbone and 30 FPS
            args.pickle_file_name = 'extraction_output_15-02-2023-18-12.pkl'
        elif args.dim_feature == 4096: # convolutional backbone
            args.pickle_file_name = 'extraction_output_22-02-2023-16-18.pkl'
        else:
            assert False, "can't assign pickle file based on simple heuristic"
    
    with open(args.dataset_file, 'r') as f:
        data_info = json.load(f)['METEOR']
    args.test_session_set = data_info['test_session_set']
    dataset_test = METEORDataLayer(phase='test', args=args)

    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    dataloader_test = DataLoader(dataset_test, 512, drop_last=False, pin_memory=True, num_workers=args.num_workers)

    # instantite model, set to eval and move to devic
    model = transformer_models.VisionTransformer_v4(args=args, img_dim=args.enc_layers,
                                                patch_dim=args.patch_dim,
                                                out_dim=args.numclass,
                                                embedding_dim=args.embedding_dim,
                                                num_heads=args.num_heads,
                                                num_layers=args.num_layers,
                                                hidden_dim=args.hidden_dim,
                                                dropout_rate=args.dropout_rate,
                                                attn_dropout_rate=args.attn_dropout_rate,
                                                num_channels=args.dim_feature,
                                                positional_encoding_type=args.positional_encoding_type,
                                                with_motion=args.use_flow
                                                    )
    model.eval()
    model.to(DEVICE)

    # load model state
    if checkpoint.lower() == 'default':
        check_str = 'checkpoint.pth'
    else:
        check_str = 'checkpoint{0:04}.pth'.format(checkpoint)
        assert check_str in os.listdir(exp_dir), f"no checkpoint nr {checkpoint} in {exp_dir}"
    
    checkpoint = torch.load(os.path.join(exp_dir, check_str), map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # instantiate y_true and y_pred
    y_true = []
    y_pred = []

    # generate predictions
    with torch.no_grad():
        for batch in tqdm(dataloader_test, desc=exp_dir):
            camera_inputs, motion_inputs, enc_target, distance_target, class_h_target, dec_target = batch
            # move inputs to device
            camera_inputs = camera_inputs.to(DEVICE)
            motion_inputs = motion_inputs.to(DEVICE)

            # generate predictions
            out = model(camera_inputs, motion_inputs)
            out = out[0] # out[1] is decoder target
            out.to('cpu').detach()
            y_true.append(class_h_target)
            y_pred.append(out)
            del camera_inputs, motion_inputs, enc_target, distance_target, class_h_target, dec_target, out

        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
        y_pred = (y_pred > 0).astype(int)
    
    # run eval metrics:
    conf_matrices = get_multilabel_conf_mat(y_true, y_pred, args.all_class_name, save_loc=exp_dir if save_fig else '')
    
    # iterate through dimensions and calculate measures per category
    metrics = {
        'precision':[],
        'recall':[],
        'f1':[],
        'roc_auc':roc_auc_score(y_true, y_pred),
        'coverage_error':coverage_error(y_true, y_pred),
        'label_ranking_avg_precision':label_ranking_average_precision_score(y_true, y_pred),
        'confusion matrices':conf_matrices
    }

    for i, class_name in enumerate(args.all_class_name):
        cyt = y_true[:,i]
        cyp = y_pred[:,i]

        metrics['precision'].append(precision_score(cyt, cyp, zero_division=0))
        metrics['recall'].append(recall_score(cyt, cyp, zero_division=0))
        metrics['f1'].append(f1_score(cyt, cyp, zero_division=0))

    if save_results:
    # write all these measures + additional information (tested epoch,...) to .txt file
        with open(os.path.join(exp_dir, 'eval.txt'), 'w') as f:
            f.write('precision :\n')
            for i, class_name in enumerate(args.all_class_name):
                f.write(f'\t{class_name}: {metrics["precision"][i]}\n')
                
            f.write('\nrecall:\n')
            for i, class_name in enumerate(args.all_class_name):
                f.write(f'\t{class_name}: {metrics["recall"][i]}\n')
                
            f.write('\nf1 score:\n')
            for i, class_name in enumerate(args.all_class_name):
                f.write(f'\t{class_name}: {metrics["f1"][i]}\n')
                
            f.write(f'\nroc auc score: {metrics["roc_auc"]}\n')
            f.write(f'coverage error: {metrics["coverage_error"]}\n')
            f.write(f'label ranking avg precision: {metrics["label_ranking_avg_precision"]}\n')

            for i, class_name in enumerate(args.all_class_name):
                f.write(f'{class_name}:\n{metrics["confusion matrices"][i]}\n')
    return metrics


def compare_experiments(root_dir, labels = ['OverTaking', 'LaneChange', 'WrongLane', 'Cutting'], save=True):
    # iterate through root dir. Per experiment load eval.txt file (if there; generate otherwise)
    # load measures into dataframe --> is to be written (if save) to root_dir and returned   
    metrics_per_cat = ['precision', 'recall', 'f1']
    
    results = pd.DataFrame(
        columns = [f'{i[0]}_{i[1]}' for i in itertools.product(metrics_per_cat, labels)] + ['roc_auc', 'coverage_error', 'label_ranking_avg_precision']
            )
    
    for root, subdirs, files in sorted(os.walk(root_dir)):
        if 'checkpoint.pth' in files:
            # print(root)
            try:
                args = get_args(root)
                dir_metrics = eval_experiment(root, args)
            except:
                continue
            # set_trace()

            # some metrics are per category --> need to be transformed
            for met_name in metrics_per_cat:
                for lab, val in zip(labels, dir_metrics[met_name]):
                    dir_metrics[met_name + '_' + lab] = val
                del dir_metrics[met_name]

            del dir_metrics['confusion matrices']
            dir_name = root.removeprefix(os.path.join(root_dir, ''))

            results.loc[dir_name,:] = dir_metrics

        else:
            continue
            
    if save:
        # TODO: one has to go
        # write one version thats machien readable
        results.to_csv(os.path.join(root_dir, 'results.csv'))

        # write human readable version too
        with open(os.path.join(root_dir, 'results_readable.txt'), 'w') as f:
            f.write(results.to_string())
    
    return results

def add_model_eval_to_comparison(exp_dir, comparison_csv_path='experiments/results.csv'):
    args = get_args(exp_dir)
    model_results = eval_experiment(exp_dir, args)
    old_df = pd.read_csv(comparison_csv_path, index_col=0)
    new_df = pd.DataFrame(columns = old_df.columns)
    metrics_per_cat = ['precision', 'recall', 'f1']
    for met_name in metrics_per_cat:
        for lab, val in zip(args.all_class_name, model_results[met_name]):
            model_results[met_name + '_' + lab] = val
        del model_results[met_name]
    del model_results['confusion matrices']
    dir_name = exp_dir.removeprefix('experiments/')
    new_df.loc[dir_name,:] = model_results
    out_df = pd.concat([old_df, new_df])
    out_df = out_df.sort_index()
    out_df.to_csv(comparison_csv_path)


def pure_sample_classification(exp_dir, save=True, **kwargs):
    # evaluate a single experiment with the provided data.
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load model config from exp_dir
    args = get_args(exp_dir)

    # prepare dataloader:
    if 'pickle_file_name' not in args.__dict__.keys():
        if args.dim_feature == 2048 and '30_FPS' not in exp_dir: # attention based backbone, not 30 FPS
            args.pickle_file_name = 'extraction_output_11-02-2023-18-33.pkl'
        elif args.dim_feature == 2048 and '30_FPS' in exp_dir: # attention based backbone and 30 FPS
            args.pickle_file_name = 'extraction_output_15-02-2023-18-12.pkl'
        elif args.dim_feature == 4096: # convolutional backbone
            args.pickle_file_name = 'extraction_output_22-02-2023-16-18.pkl'
        else:
            assert False, "can't assign pickle file based on simple heuristic"
    
    with open(args.dataset_file, 'r') as f:
        data_info = json.load(f)['METEOR']
    args.test_session_set = data_info['test_session_set']

    dataset_test = PURE_METEORDataLayer(phase='test', args=args)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    dataloader_test = DataLoader(dataset_test, 512, drop_last=False, pin_memory=True, num_workers=args.num_workers)

   
    # instantite model, set to eval and move to devic
    model = transformer_models.VisionTransformer_v3(args=args, img_dim=args.enc_layers,
                                                patch_dim=args.patch_dim,
                                                out_dim=args.numclass,
                                                embedding_dim=args.embedding_dim,
                                                num_heads=args.num_heads,
                                                num_layers=args.num_layers,
                                                hidden_dim=args.hidden_dim,
                                                dropout_rate=args.dropout_rate,
                                                attn_dropout_rate=args.attn_dropout_rate,
                                                num_channels=args.dim_feature,
                                                positional_encoding_type=args.positional_encoding_type,
                                                with_motion=args.use_flow
                                                    )
    model.eval()
    model.to(DEVICE)
    
    # load model state
        
    check_str = kwargs.get('checkpoint', 'checkpoint.pth')
    if check_str != 'checkpoint.pth':
        check_str = 'checkpoint{0:04}.pth'.format(check_str)
        assert check_str in os.listdir(exp_dir), f"no checkpoint {check_str} in {exp_dir}"
    
    checkpoint = torch.load(os.path.join(exp_dir, check_str), map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # instantiate y_true and y_pred
    y_true = []
    y_pred = []

    # generate predictions
    with torch.no_grad():
        for batch in tqdm(dataloader_test, desc=exp_dir):
            camera_inputs, motion_inputs, enc_target, distance_target, class_h_target, dec_target = batch
            # move inputs to device
            camera_inputs = camera_inputs.to(DEVICE)
            motion_inputs = motion_inputs.to(DEVICE)

            # generate predictions
            out = model(camera_inputs, motion_inputs)
            out = out[0] # out[1] is decoder target
            out.to('cpu').detach()
            y_true.append(class_h_target)
            y_pred.append(out)
            del camera_inputs, motion_inputs, enc_target, distance_target, class_h_target, dec_target, out
        
        # concatenate, move to cpu and transform to numpy
        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
        y_pred = (y_pred > 0).astype(int)
    
    class_more_than_one = np.where(np.sum(y_pred, axis=1) > 1)[0]
    print(f'{len(class_more_than_one)} samples were classifyied as more than one category')
    
    out_df = pd.DataFrame(columns = ['total_examples', 'nothing'] + args.all_class_name, index = args.all_class_name)
    
    for dim, name in enumerate(args.all_class_name):
        # set_trace()
        
        # get dimensions from y_true that correspond to pure examples of that category
        dim_i = np.where(y_true[:,dim] == 1)[0]
        
        # select these dimensions from y_pred 
        dim_y = y_pred[dim_i]
        
        # get number of examples from y_true
        examples = len(dim_i)

        # get nr. of rows that were not assigned to any class
        nothing = np.where(np.sum(dim_y, axis=1) == 0)[0].shape[0]
        
        # get rest
        rest = np.sum(dim_y, axis=0)
        rest = np.expand_dims(rest, 0)
        
        set_trace()
        # concat data 
        dim_data = np.column_stack([np.array([examples]), np.array([nothing]), rest])
        
        # add to data frame
        out_df.loc[name,:] = dim_data
        
    out_file_name = 'pure_classification.csv'
    
    out_df.to_csv(os.path.join(exp_dir, out_file_name))
    
    
    
    
    
    
if __name__ == '__main__':
    # add_model_eval_to_comparison('experiments/att_back/2_unsc_loss_dropout01')
    add_model_eval_to_comparison('experiments/att_back/new_loss_feat_drop')
    