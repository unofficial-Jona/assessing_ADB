import os

import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace

from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, f1_score, precision_score, recall_score, label_ranking_average_precision_score, coverage_error


from misc import utils


all_class_name = [
    "Background",
    "OverTaking",
    "LaneChange",
    "WrongLane",
    "Cutting"
    ]


def Colar_evaluate(results, epoch, command, log_file):
    map, aps, cap, caps = utils.frame_level_map_n_cap(results)
    out = '[Epoch-{}] [IDU-{}] mAP: {:.4f}\n'.format(epoch, command, map)
    # print(out)
    
    if log_file != '':
        with open(log_file, 'a+') as f:
            f.writelines(out)
        for i, ap in enumerate(aps):
            cls_name = all_class_name[i]
            out = '{}: {:.4f}\n'.format(cls_name, ap)
            print(out)
            with open(log_file, 'a+') as f:
                f.writelines(out)

def get_multilabel_conf_mat(y_true, y_pred, save_loc='', visualize=False):
    """ convenience wrapper for skelarn.metrics.multilabel_confusion_matrix. Can visualize and 

    Args:
        y_true (np.array): Ground Truth
        y_pred (np.array): Predictions
        save (str, optional): If set saves the plot at the given location. Defaults to '', meaning no save.
        visualize (bool, optional): Whether to visualize the output. Defaults to False.

    Returns:
        np.array: confusion matrices as returned by sklearn.metrics.multilabel_confusion_matrix
    """
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(confusion_matrices.shape[0], 1, figsize=(15,15))
                        
    for i, (mat, lab) in enumerate(zip(confusion_matrices, all_class_name)):
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
                
                
def evaluate_running_experiment(results, log_file):

    location = os.path.join(*log_file.split('/')[:-1])
    assert os.path.exists(location)
    
    y_true = results['labels']
    y_pred = results['probs']
    assert y_pred.min() >= 0 and y_pred.max() <= 1, 'predicted probabilites are not in range [0,1]'
    
    y_pred_cls = np.round(y_pred)
    
    get_multilabel_conf_mat(y_true.T, y_pred_cls.T, save_loc=location)
    
    metrics = {
        'roc_auc_score': roc_auc_score(y_true, y_pred_cls),
        'weighted_f1': f1_score(y_true, y_pred_cls, average='weighted'),
        'weighted_precision': precision_score(y_true, y_pred_cls, average='weighted'), 
        'weighted_recall': recall_score(y_true, y_pred_cls, average='weighted')
    }

    with open(os.path.join(location, 'evaluation.txt'), 'a') as f:
        for k, v in metrics.items():
            str_out = f'{k}: {v}\n'
            f.write(str_out)
                
if __name__ == '__main__':
    pass