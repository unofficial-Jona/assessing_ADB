import os
import datetime
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import loss as utl
from misc import init as cfg
# from Dataload.ThumosDataset import THUMOSDataSet
from Dataload.MeteorDataset import MeteorDataset
from loss.evaluate import Colar_evaluate, evaluate_running_experiment
from model.ColarModel import Colar_dynamic, Colar_static, CombinedColar
import torch.nn.functional as F
import numpy as np
from misc.utils import backup_code
from misc.custom_utils import save_args, evaluate_save_results

from pdb import set_trace

def train_one_epoch(model, criterion, data_loader, optimizer, device, max_norm):
    model.train()
    criterion.train()
    
    losses = 0
    i = 0
    
    batch_num = len(iter(data_loader))
    
    for camera_inputs, enc_target in data_loader:
        inputs = camera_inputs.to(device)
        target = enc_target.to(device)
        
        optimizer.zero_grad()
        
        out, dynamic_out, static_out = model(inputs)
        
        loss_static = criterion(static_out.squeeze(), target[:, -1:].squeeze(), 'ML')
        loss_dynamic = criterion(dynamic_out[:,:,-1:].squeeze(), target[:, -1:].squeeze(), 'ML')
        loss_network = criterion(out.squeeze(), target[:,-1:].squeeze(), 'ML')
        loss_KL = criterion(dynamic_out[:,:,-1:], static_out, 'KL_new')
        
        loss = loss_static + loss_dynamic + loss_network + loss_KL
        
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        losses += loss
        i = i + 1
        print('\r train-------------------{:.4f}%'.format((i / batch_num) * 100), end='')
    return losses / i, losses

def evaluate(model, data_loader, device):
    model.eval()

    score_val_x = []
    target_val_x = []

    i = 0
    batch_num = len(iter(data_loader))
    for camera_inputs, enc_target in data_loader:
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
        i += 1
        print('\r test--------------------{:.4f}%'.format((i / batch_num) * 100), end='')
    all_probs = np.asarray(score_val_x).T
    all_classes = np.asarray(target_val_x).T
    # print(all_probs.shape, all_classes.shape)
    results = {'probs': all_probs, 'labels': all_classes}
    return results


def main(args):
    log_file = backup_code(args.exp_name)
    # log_file = 'output/log.txt'
    seed = args.seed + cfg.get_rank()
    cfg.set_seed(seed)
    
    save_args(args, log_file)

    device = torch.device('cuda:' + str(args.cuda_id))
    
    dataset_train = MeteorDataset(flag='train', args=args, weights = args.use_weights)
    dataset_val = MeteorDataset(flag='test', args=args)

    loss_weights = dataset_train.weights
    
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   pin_memory=True, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, pin_memory=True, num_workers=args.num_workers)
    
    model = CombinedColar(args.input_size, args.numclass, device, args.kmean)
    
    model.apply(cfg.weight_init)
    model.to(device)
    
    criterion = utl.SetCriterion(weights=loss_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, loss = train_one_epoch(model, criterion, data_loader_train, optimizer, device, args.clip_max_norm)

        lr_scheduler.step()
        print('\nepoch:{}------loss:{}'.format(epoch, train_loss))
        
        test_stats = evaluate(model, data_loader_val, device)
        print('\n---------------Calculation of the map-----------------')
        Colar_evaluate(test_stats, epoch, args.command, log_file)
        
        if (epoch % 50 == 0 and epoch != 0) or epoch + 1 == args.epochs:
            file_names = log_file.split('/')
            
            torch.save(model.state_dict(), os.path.join(*file_names[:-1], f'checkpoint_{epoch}.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    print('evaluating experiment')
    evaluate_save_results(test_stats, log_file)

if '__main__' == __name__:
    args = cfg.parse_args()
    with open(args.dataset_file, 'r') as f:
        data_info = json.load(f)['METEOR']

    args.train_session_set = data_info['train_session_set']
    args.test_session_set = data_info['test_session_set']
    args.class_index = data_info['class_index']
    
    args.kmean = '/workspace/pvc-meteor/features/colar/gmm_centers.pickle'
    for weights in ['all', 'recent']:
        args.use_weights = weights
        main(args)
