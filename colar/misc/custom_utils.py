import warnings
import os
import pickle
from model.ColarModel import Colar_dynamic, Colar_static
from Dataload.MeteorDataset import MeteorDataset
from misc.init import parse_args
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

def get_multilabel_conf_mat(y_true, y_pred, label_names=["Background", "OverTaking", "LaneChange", "WrongLane", "Cutting"], save_loc='', visualize=False):
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

def evaluate(model_dynamic,
             model_static,
             data_loader, device):
    model_static.eval()
    model_dynamic.eval()

    score_val_x = []
    target_val_x = []

    i = 0
    batch_num = len(iter(data_loader))
    for camera_inputs, enc_target in data_loader:
        inputs = camera_inputs.to(device)
        target = enc_target.to(device)
        target_val = target[:, -1:, :5]

        with torch.no_grad():
            enc_score_dynamic = model_dynamic(inputs)
            enc_score_static = model_static(inputs[:, -1:, :], device)
        
        enc_score_static = enc_score_static.permute(0, 2, 1)
        enc_score_static = enc_score_static[:, :, :5]

        enc_score_dynamic = enc_score_dynamic.permute(0, 2, 1)
        enc_score_dynamic = enc_score_dynamic[:, -1:, :5]

        score_val = enc_score_static * 0.3 + enc_score_dynamic * 0.7
        score_val = F.sigmoid(score_val)

        score_val = score_val.contiguous().view(-1, 5).cpu().numpy()
        target_val = target_val.contiguous().view(-1, 5).cpu().numpy()

        score_val_x += list(score_val)
        target_val_x += list(target_val)
        i += 1
        print('\r test--------------------{:.4f}%'.format((i / batch_num) * 100), end='')
    all_probs = np.asarray(score_val_x).T
    all_classes = np.asarray(target_val_x).T
    # print(all_probs.shape, all_classes.shape)
    results = {'probs': all_probs, 'labels': all_classes}
    return results

def evaluate_exp(exp_root):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = load_args(exp_root)
    
    model_dynamic = Colar_dynamic(args.input_size, args.numclass)
    model_static = Colar_static(args.input_size, args.numclass, DEVICE, args.kmean)
    
    # load model checkpoints
    dynamic_path, static_path = get_checkpoint_name(exp_root)
    model_dynamic.load_state_dict(torch.load(os.path.join(exp_root, dynamic_path)))
    model_static.load_state_dict(torch.load(os.path.join(exp_root, static_path)))

    # set to eval() and move to device
    model_dynamic = model_dynamic.eval().to(DEVICE)
    model_static = model_static.eval().to(DEVICE)

    # set up dataset and run inference
    dataset = MeteorDataset(flag='test', args=args)
    sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = DataLoader(dataset, args.batch_size, sampler=sampler, drop_last=False, pin_memory=True, num_workers=args.num_workers)
    
    with torch.no_grad():
        results = evaluate(model_dynamice, model_static, data_loader, DEVICE)

    rounded_results = np.round(results['probs'])
    ground_truth = results['labels']

    get_multilabel_conf_mat(y_true = ground_truth, y_pred = rounded_results, save_fig = exp_root)

    metrics = {
        'roc_auc_score': roc_auc_score(ground_truth, rounded_results),
        'weighted_f1': f1_score(ground_truth, rounded_results, average='weighted'),
        'weighted_precision': precision_score(ground_truth, rounded_results, average='weighted'), 
        'weighted_recall': recall_score(ground_truth, rounded_results, average='weighted')
    }

    with open(os.path.join(exp_root, 'evaluation.txt'), 'a') as f:
        for k, v in metrics.items():
            str_out = f'{k}: {v}\n'
            f.write(str_out)
            
if __name__ == '__main__':
    exp_name = 'output/2023_04_12_03_10_48_ColarMETEOR'
    evaluate_exp(exp_name)