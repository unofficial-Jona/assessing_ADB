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
from loss.evaluate import Colar_evaluate
from model.ColarModel import Colar_dynamic, Colar_static
import torch.nn.functional as F
import numpy as np
from misc.utils import backup_code
from misc.custom_utils import save_args, evaluate_exp

from pdb import set_trace


def train_one_epoch(model_dynamic,
                    model_static,
                    criterion,
                    data_loader, optimizer,
                    device, max_norm):
    model_static.train()
    model_dynamic.train()
    criterion.train()
    losses = 0
    i = 0
    batch_num = len(iter(data_loader))
    for camera_inputs, enc_target in data_loader:
        inputs = camera_inputs.to(device)
        
        # can't keep argmax for multilabel classification
        # enc_target = enc_target.argmax(dim=-1)
        target = enc_target.to(device=device)

        optimizer.zero_grad()

        enc_score_static = model_static(inputs[:, -1:, :], device)
        loss_static = criterion(enc_score_static.squeeze(), target[:, -1:].squeeze(), 'ML')
        
        enc_score_dynamic = model_dynamic(inputs)
        loss_dynamic = criterion(enc_score_dynamic[:, :, -1:].squeeze(), target[:, -1:].squeeze(), 'ML')

        loss_KL = criterion(enc_score_dynamic[:, :5, -1:], enc_score_static, 'KL_new')
        loss = loss_static + loss_dynamic + loss_KL

        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model_static.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(model_dynamic.parameters(), max_norm)

        optimizer.step()
        losses += loss
        i = i + 1
        print('\r train-------------------{:.4f}%'.format((i / batch_num) * 100), end='')
    return losses / i, losses

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


def main(args):
    log_file = backup_code(args.exp_name)
    seed = args.seed + cfg.get_rank()
    cfg.set_seed(seed)

    device = torch.device('cuda:' + str(args.cuda_id))
    model_static = Colar_static(args.input_size, args.numclass, device, args.kmean)
    model_dynamic = Colar_dynamic(args.input_size, args.numclass)

    model_dynamic.apply(cfg.weight_init)
    model_dynamic.to(device)
    model_static.apply(cfg.weight_init)
    model_static.to(device)

    save_args(args, log_file)
    
    criterion = utl.SetCriterion().to(device)
    optimizer = torch.optim.Adam([
        {"params": model_static.parameters()},
        {"params": model_dynamic.parameters()}],
        lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = MeteorDataset(flag='train', args=args)
    dataset_val = MeteorDataset(flag='test', args=args)
    
    # set_trace() 
    
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   pin_memory=True, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, pin_memory=True, num_workers=args.num_workers)

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, loss = train_one_epoch(
            model_dynamic,
            model_static,
            criterion, data_loader_train, optimizer, device, args.clip_max_norm)

        lr_scheduler.step()
        print('\nepoch:{}------loss:{}'.format(epoch, train_loss))
        
        test_stats = evaluate(
            model_dynamic,
            model_static,
            data_loader_val, device)
        print('\n---------------Calculation of the map-----------------')
        Colar_evaluate(test_stats, epoch, args.command, log_file)
        
        if (epoch % 50 == 0 and epoch != 0) or epoch + 1 == args.epochs:
            file_names = log_file.split('/')
            
            torch.save(model_static.state_dict(), os.path.join(*file_names[:-1], f'static_{epoch}.pth'))
            torch.save(model_dynamic.state_dict(), os.path.join(*file_names[:-1], f'dynamic_{epoch}.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    print('evaluating experiment')
    evaluate_exp(os.path.join(*file_names[:-1]))


if '__main__' == __name__:
    args = cfg.parse_args()
    with open(args.dataset_file, 'r') as f:
        data_info = json.load(f)['METEOR']

    args.train_session_set = data_info['train_session_set']
    args.test_session_set = data_info['test_session_set']
    args.class_index = data_info['class_index']
    
    args.output_dir = 'experiment/init/'
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
