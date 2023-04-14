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
import warnings

from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, f1_score, precision_score, recall_score, label_ranking_average_precision_score, coverage_error

from custom_dataset import METEORDataLayer, PURE_METEORDataLayer, METEOR_3D
import transformer_models

from pdb import set_trace

all_label_names=['Background', 'OverTaking', 'LaneChange', 'WrongLane', 'Cutting']
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
    args = ModelConfig(**dic)
    
    data_info = json.load(open(args.dataset_file, 'r'))['METEOR']
    args.train_session_set = data_info['train_session_set']
    args.test_session_set = data_info['test_session_set']
    return args

def get_model(args, model_flag='v3'):
    if model_flag == 'v3':
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
                                                ).eval()

    elif model_flag == 'v4':
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
                                                ).eval()
    else:
        assert False, 'invalid model flag'
    return model

def load_model_and_checkpoint(args, directory, checkpoint='default'):
    if checkpoint.lower() == 'default':
        check_str = 'checkpoint.pth'
    else:
        check_str = 'checkpoint{0:04}.pth'.format(checkpoint)
        assert check_str in os.listdir(directory), f"no checkpoint nr {checkpoint} in {exp_dir}"

    checkpoint = torch.load(os.path.join(directory, check_str), map_location='cpu')
    
    # try loading checkpoint into both models.
    try: 
        model = get_model(args)
        model.load_state_dict(checkpoint['model'])
    except:
        model = get_model(args, 'v4')
        model.load_state_dict(checkpoint['model'])
    return model

def get_predictions(model, dataset, device='cuda', batch_size=512):
    if not model.is_cuda:
        model = model.to('cuda')
    
    model = model.eval()
    
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = DataLoader(dataset, batch_size, drop_last=False, pin_memory=True)
    
    y_true, y_pred = list(), list()
    for batch in data_loader:
        camera_inputs, motion_inputs, _, _, class_h_target, _ = batch
        camera_inputs = camera_inputs.to(device)
        motion_inputs = motion_inputs.to(device)
        
        out = model(camera_inputs, motion_inputs)
        out = out[0].to('cpu').detach()
        
        y_pred.append(out)
        y_true.append(class_h_target)
        
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    y_pred = (y_pred > 0).astype(int)
    
    return y_true, y_pred

def get_weighted_metrics(y_true, y_pred, save_loc='', **kwargs):
    metrics=[roc_auc_score, f1_score, precision_score, recall_score]
    metric_names = ['weighted_roc_auc', 'weighted_f1', 'weighted_precision', 'weighted_recall']
    
    epoch = kwargs.get('epoch','')
    
    out_dict = dict()
    for name, metric in zip(metric_names, metrics):
        value = metric(y_true, y_pred, average='weighted')
        out_dict.update({name: value})
    
    if save_loc:
        with open(os.path.join(save_loc, 'evaluation.txt'), 'a') as f:
            if epoch:
                f.write('\n')
                f.write(f'epoch: {epoch}')
            for k, v in out_dict.items():
                f.write(f'{k}: {v}\n')
            f.write('\n')
    return out_dict

def get_metric_per_category(y_true, y_pred, label_names=all_label_names, save_loc='', **kwargs):
    metrics = [f1_score, precision_score, recall_score]
    metric_names = ['f1', 'precision', 'recall']
    
    epoch = kwargs.get('epoch','')
    
    out_dict = dict()
    for i, label_name in enumerate(label_names):
        label_dict = dict()
        y_true_label = y_true[:,i]
        y_pred_label = y_pred[:,i]
        
        for metric, metric_name in zip(metrics, metric_names):
            value = metric(y_true_label, y_pred_label)
            label_dict.update({f'{label_name}_{metric_name}': value})
        out_dict = {**out_dict, **label_dict}

    assert len(out_dict.keys()) == len(metric_names) * len(label_names), 'out dict seems wrong'
    
    if save_loc:           
        with open(os.path.join(save_loc, 'evaluation.txt'), 'a') as f:
            if epoch:
                f.write('\n')
                f.write(f'epoch: {epoch}')
            for k, v in out_dict.items():
                f.write(f'{k}: {v}\n')
            f.write('\n')
    
    return out_dict    

def get_multilabel_conf_mat(y_true, y_pred, label_names=all_label_names, save_loc='', visualize=False, **kwargs):
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
    
    epoch = kwargs.get('epoch', '')
                        
    for i, (mat, lab) in enumerate(zip(confusion_matrices, label_names)):
        disp = ConfusionMatrixDisplay(mat, display_labels=[lab + '-','+'+lab])
        disp.plot(ax=ax[i], values_format='.0f', cmap='Blues', xticks_rotation='horizontal')
        disp.ax_.set_title(lab)
        disp.im_.colorbar.remove()
    plt.tight_layout()
    if visualize:
        plt.show()
    if save_loc:
        file_name = 'confusion_matrices'
        if epoch:
            file_name += f'_{epoch}'
        plt.savefig(os.path.join(save_loc, file_name + '.png')) #confusion matrices
    return confusion_matrices

def evaluate_experiment(directory, **kwargs):
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    save_results = kwargs.get('save_results', True)
    save_fig = kwargs.get('save_fig', True)
    
    args = get_args(directory)
    model = load_model_and_checkpoint(args, directory)
    if 'v4' in directory:
        dataset = METEOR_3D(args, 'test')
    else:
        dataset = METEORDataLayer(args, 'test')
   
    with torch.no_grad():
        y_true, y_pred = get_predictions(model, dataset, device)
    
    if y_pred.shape[1] == 4:
        # only 4 most recent categories --> cut from y_true
        y_true = y_true[:,1:]
        if not hasattr(args, 'all_class_name'):
            args.all_label_name=all_label_names[1:]            
    elif y_pred.shape[1] > 5:
        # all 7 categories --> use index [0,2,4,6] in y_pred and cut first dim from y_true
        y_pred = y_pred[:,[0,2,4,6]]
        y_true = y_true[:,1:]
        if not hasattr(args, 'all_class_name'):
            args.all_label_name=all_label_names[1:]        
    elif y_pred.shape[1] == 5:
        # 4 most recent + background --> same as y_true
        if not hasattr(args, 'all_class_name'):
            args.all_label_name = all_label_names
        pass
    else:
        assert False, f'unrecognized shape for y_pred: {y_pred.shape}'
    
    assert y_pred.shape[1] == len(args.all_class_name), 'prediction shape and label names are off' 
    assert y_true.shape == y_pred.shape, 'label and prediction shape do not match'
    
    get_multilabel_conf_mat(y_true, y_pred, label_names=args.all_class_name, save_loc=directory if save_fig else '', **kwargs)
    cat_metrics = get_metric_per_category(y_true, y_pred, label_names=args.all_class_name, save_loc=directory if save_results else '', **kwargs)
    weighted_metrics = get_weighted_metrics(y_true, y_pred, save_loc=directory if save_results else '', **kwargs)
    
    return {**weighted_metrics, **cat_metrics}

def add_model_eval_to_comparison(exp_dir, comparison_csv_path='experiments/results.csv'):
    returned_metrics = evaluate_experiment(exp_dir)
    exp_df = pd.DataFrame([returned_metrics.values()], columns = returned_metrics.keys(), index = [exp_dir.removeprefix('experiments/')])
    
    # check if comparion file exists, if so append experiment to it
    if os.path.exists(comparison_csv_path):
        old_df = pd.read_csv(comparison_csv_path, index_col=0)
        out_df = pd.concat([old_df, exp_df])
    else:
        out_df = exp_df
    out_df.to_csv(comparison_csv_path)

def compare_experiments(root_dir):
    for root, subdirs, files in tqdm(sorted(os.walk(root_dir))):
        if 'checkpoint.pth' in files:
            add_model_eval_to_comparison(root)


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
    # add_model_eval_to_comparison('experiments/att_back/new_loss_feat_drop')
    compare_experiments('experiments')