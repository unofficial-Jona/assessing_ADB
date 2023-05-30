import warnings
import os
import pickle
from model.ColarModel import Colar_dynamic, Colar_static, CombinedColar
from Dataload.MeteorDataset import MeteorDataset
from misc.init import parse_args
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, f1_score, precision_score, recall_score, label_ranking_average_precision_score, coverage_error

from pdb import set_trace

class ModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

def save_args(args, save_loc):
    if save_loc.endswith('.txt'):
        save_loc = os.path.join(*save_loc.split('/')[:-1])

    out_dict = dict()
    for arg in vars(args):
        out_dict.update({arg:getattr(args, arg)})

    out_name = os.path.join(save_loc, 'args.pkl')
    pickle.dump(out_dict, open(out_name, 'wb'))

def load_args(exp_root):
    try:
        load_obj = pickle.load(open(os.path.join(exp_root, 'args.pkl'), 'rb'))
        return ModelConfig(**load_obj)
    except:
        warnings.warn(f'could not find args.pkl in {exp_root}, loading args from init file instead')
        return parse_args()

def get_checkpoint_name(exp_root, epoch='DEFAULT'):
    checkpoint_list = [i for i in os.listdir(exp_root) if i.endswith('.pth')]

    dynamic = [i for i in checkpoint_list if i.startswith('dynamic')]
    static = [i for i in checkpoint_list if i.startswith('static')]

    dynamic.sort()
    static.sort()

    if epoch == 'DEFAULT':
        checkpoint_dynamic = dynamic[-1]
        checkpoint_static = static[-1]

    else:
        checkpoint_dynamic = f'dynamic_{epoch}.pth'
        checkpoint_static = f'static_{epoch}.pth'

    return checkpoint_dynamic, checkpoint_static

def get_multilabel_conf_mat(results, label_names=["Background", "OverTaking", "LaneChange", "WrongLane", "Cutting"], save_loc='', visualize=False):
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
    for k, v in results.items():
        assert v.shape[0] > v.shape[1], f'result[{k}] has wrong shape'
    
    y_true = results['labels']
    y_pred = np.round(results['probs'])
    
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

def load_combined_colar_and_checkpoint(args, exp_dir, checkpoint='default', **kwargs):
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_list = glob(os.path.join(exp_dir, '*.pth'))
    if checkpoint.lower() == 'default':
        checkpoint_path = sorted(checkpoint_list)[-1]
    else:
        checkpoint_path = f'checkpoint_{checkpoint}.pth'
        assert os.path.exists(checkpoint_path), f'no file named checkpoint_{checkpoint}.pth in {exp_dir}'
    
    model = CombinedColar(args.input_size, args.numclass, device, args.kmean)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def load_old_colar_and_checkpoint(args, exp_dir, checkpoint='default', **kwargs):
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    dyn_checkpoint_list = glob(os.path.join(exp_dir, 'static_*.pth'))
    stat_checkpoint_list = glob(os.path.join(exp_dir, 'dynamic_*.pth'))
    if checkpoint.lower() == 'default':
        dyn_checkpoint = sorted(dyn_checkpoint_list)[-1]
        stat_checkpoint = sorted(stat_checkpoint_list)[-1]
    else:
        dyn_checkpoint = os.path.join(exp_dir, f'dynamic_{checkpoint}.pth')
        stat_checkpoint = os.path.join(exp_dir, f'static_{checkpoint}.pth')
        assert os.path.exists(dyn_checkpoint), f'no file named checkpoint_{checkpoint}.pth in {exp_dir}'
        assert os.path.exists(stat_checkpoint), f'no file named checkpoint_{checkpoint}.pth in {exp_dir}'
    
    model_static = Colar_static(args.input_size, args.numclass, device, args.kmean)
    model_dynamic = Colar_dynamic(args.input_size, args.numclass)

    model_static.eval()
    model_dynamic.eval()
    return model_static, model_dynamic

def get_test_loader(args):
    dataset = MeteorDataset(flag='test', args=args)
    sampler = torch.utils.data.SequentialSampler(dataset)
    return DataLoader(dataset, 512, sampler=sampler, drop_last=False, pin_memory=True, num_workers=4)

def get_results_old_colar(exp_dir, **kwargs):
    args = load_args(exp_dir)
    
    if 'model_static' in kwargs.keys() and 'model_dynamic' in kwargs.keys():
        model_static = kwargs['model_static']
        model_dynamic = kwargs['model_dynamic']
    else:
        model_static, model_dynamic = load_old_colar_and_checkpoint(args, exp_dir)
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model_static = model_static.to(device)
    model_dynamic = model_dynamic.to(device)

    dataloader = kwargs.get('dataloader', get_test_loader(args))

    score_val_x = []
    target_val_x = []

    for camera_inputs, enc_target in dataloader:
        inputs = camera_inputs.to(device)
        target = enc_target.to(device)
        target_val = target[:, -1:, :5]

        with torch.no_grad():
            enc_score_dynamic = model_dynamic(inputs)
            enc_score_static = model_static(inputs[:, -1:, :])
        
        enc_score_static = enc_score_static.permute(0, 2, 1)
        enc_score_static = enc_score_static[:, :, :5]

        enc_score_dynamic = enc_score_dynamic.permute(0, 2, 1)
        enc_score_dynamic = enc_score_dynamic[:, -1:, :5]

        score_val = enc_score_static * 0.3 + enc_score_dynamic * 0.7
        score_val = torch.sigmoid(score_val)

        score_val = score_val.contiguous().view(-1, 5).cpu().numpy()
        target_val = target_val.contiguous().view(-1, 5).cpu().numpy()

        score_val_x += list(score_val)
        target_val_x += list(target_val)
    all_probs = np.asarray(score_val_x).T
    all_classes = np.asarray(target_val_x).T

    results = {'probs': all_probs, 'labels': all_classes}
    return results

def get_results_new_colar(exp_dir, **kwargs):
    args = load_args(exp_dir)
    model = kwargs.get('model', load_combined_colar_and_checkpoint(args, exp_dir))
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    dataloader = kwargs.get('dataloader', get_test_loader(args))

    score_val_x = []
    target_val_x = []
    
    for camera_inputs, enc_target in dataloader:
        inputs = camera_inputs.to(device)
        target = enc_target.to(device)
        target_val = target[:, -1:, :5]

        with torch.no_grad():
            out, _, _ = model(inputs)
        
        score_val = torch.sigmoid(out)

        score_val = score_val.contiguous().view(-1, 5).cpu().numpy()
        target_val = target_val.contiguous().view(-1, 5).cpu().numpy()

        score_val_x += list(score_val)
        target_val_x += list(target_val)
    
    all_probs = np.asarray(score_val_x).T
    all_classes = np.asarray(target_val_x).T

    results = {'probs': all_probs, 'labels': all_classes}
    return results

def get_weighted_metrics(results, **kwargs):
    for k, v in results.items():
        assert v.shape[0] > v.shape[1], f'result[{k}] has wrong shape'
    
    rounded_results = np.round(results['probs'])
    ground_truth = results['labels']

    metrics = {
        'roc_auc_score': roc_auc_score(ground_truth, rounded_results),
        'weighted_f1': f1_score(ground_truth, rounded_results, average='weighted'),
        'weighted_precision': precision_score(ground_truth, rounded_results, average='weighted'), 
        'weighted_recall': recall_score(ground_truth, rounded_results, average='weighted')
    }
    return metrics

def get_individual_metrics(results, label_names=["Background", "OverTaking", "LaneChange", "WrongLane", "Cutting"]):
    for k, v in results.items():
        assert v.shape[0] > v.shape[1], f'result[{k}] has wrong shape'
        
    metrics = [roc_auc_score, f1_score, precision_score, recall_score]
    metric_names = ['roc_auc', 'f1', 'precision', 'recall']

    y_true = results['labels']
    y_pred = np.round(results['probs'])
    
    out_dict = dict()
    for i, label_name in enumerate(label_names):
        cat_y_true = y_true[:, i]
        cat_y_pred = y_pred[:, i]
        for metric, metric_name in zip(metrics, metric_names):
            value = metric(cat_y_true, cat_y_pred)
            out_dict.update({f'{label_name}_{metric_name}':value})
    return out_dict

def evaluate_save_results(results, location):
    if location.endswith('.txt'):
        location = os.path.join(*location.split('/')[:-1])
    out_res = dict()
    
    for k, v in results.items():
        if v.shape[0] < v.shape[1]:
            results[k] = v.T
    
    # save weighted metrics
    metrics = get_individual_metrics(results)
    with open(os.path.join(location, 'evaluation.txt'), 'a') as f:
        for k, v in metrics.items():
            str_out = f'{k}: {v}\n'
            f.write(str_out)
        f.write('\n')
    out_res = {**metrics}
    
    # save individual metrics
    metrics = get_weighted_metrics(results)
    with open(os.path.join(location, 'evaluation.txt'), 'a') as f:
        for k, v in metrics.items():
            str_out = f'{k}: {v}\n'
            f.write(str_out)
        f.write('\n')
    out_res = {**out_res , **metrics}
    
    # save conf matrix 
    get_multilabel_conf_mat(results, save_loc=location)
    return out_res

def evaluate_exp_dir(exp_dir, **kwargs):
    # decide if directory belongs to old or new colar
    static_checkpoints = glob(os.path.join(exp_dir, 'static_*.pth'))
    new_checkpoint = glob(os.path.join(exp_dir, 'checkpoint_*.pth'))
    if static_checkpoints:
        results = get_results_old_colar(exp_dir, **kwargs)
    elif new_checkpoint:
        results = get_results_new_colar(exp_dir, **kwargs)
    else:
        assert False, "can't load colar based on checkpoints"

    # flag to decide if file should be overwritten
    overwrite = kwargs.get('overwrite', False)

    # if file does not exist --> create it
    if not os.path.exists(os.path.join(exp_dir, 'evaluation.txt')):
        evaluate_save_results(results, exp_dir)
    
    # if file exists and overwrite == True --> create it
    elif overwrite:
        evaluate_save_results(results, exp_dir)

def evaluate_all_outputs(root_dir = 'output/', **kwargs):
    for root, subdirs, files in tqdm(sorted(os.walk(root_dir))):
        for file in files:
            if '.pth' in file:
                evaluate_exp_dir(root, **kwargs)
                break

if __name__ == '__main__':
    evaluate_all_outputs(overwrite=True)
