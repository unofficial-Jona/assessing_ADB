import argparse
import datetime
import json
import random
import time
from pathlib import Path
import warnings
# from config import get_args_parser
from custom_config import get_args_parser
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader, DistributedSampler
from ipdb import set_trace
import util as utl
import os
import utils

from custom_dataset import METEORDataLayer, METEOR_3D
from custom_utils import add_model_eval_to_comparison, generate_dict, ModelConfig, get_multilabel_conf_mat, get_metric_per_category, get_weighted_metrics, get_predictions
import transformer_models
from train import train_one_epoch, evaluate
from test import test_one_epoch
import torch.nn as nn

from torchinfo import summary


def main(args):
    utils.init_distributed_mode(args)
    command = 'python ' + ' '.join(sys.argv)
    this_dir = args.output_dir
    if args.removelog:
        if args.distributed:
            print('distributed training !')
            if utils.is_main_process():
                print('remove logs !')
                if os.path.exists(os.path.join(this_dir, 'log_dist.txt')):
                    os.remove(os.path.join(this_dir, 'log_dist.txt'))
                if os.path.exists(Path(args.output_dir) / "log_train&test.txt"):
                    os.remove(Path(args.output_dir) / "log_train&test.txt")
        else:
            print('remove logs !')
            if os.path.exists(os.path.join(this_dir, 'log_dist.txt')):
                os.remove(os.path.join(this_dir, 'log_dist.txt'))
            if os.path.exists(Path(args.output_dir) / "log_train&test.txt"):
                os.remove(Path(args.output_dir) / "log_train&test.txt")
    logger = utl.setup_logger(os.path.join(this_dir, 'log_dist.txt'), command=command)
    # logger.output_print("git:\n  {}\n".format(utils.get_sha()))

    
    # prepare data_loader
    dataset_train = METEOR_3D(phase='train', args=args, weights='recent')
    dataset_val = METEOR_3D(phase='test', args=args)
    
    
    args.weight_values = dataset_train.weights
    
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   pin_memory=True, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, pin_memory=True, num_workers=args.num_workers)

    
    # save args
    for arg in vars(args):
        if 'session_set' in arg:
            continue
        logger.output_print("{}:{}".format(arg, getattr(args, arg)))

    # set device
    if args.distributed:
        print('args.gpu : ', args.gpu)
        torch.cuda.set_device(args.gpu)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # prepare model
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
                                                 with_motion=False
                                                   )
    
    model.to(device)

    summary(model)
    
    # prepare loss
    loss_need = [
        'labels_encoder',
        'labels_decoder',
    ]
    
    criterion = utl.SetCriterion(num_classes=args.numclass, losses=loss_need, args=args).to(device)

    # set up distribution and 
    model_without_ddp = model
    if args.distributed:
        # torch.cuda.set_device(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    elif args.dataparallel:
        args.gpu = '0,1,2,3'
        model = nn.DataParallel(model, device_ids=[int(iii) for iii in args.gpu.split(',')])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.output_print('number of params: {}'.format(n_parameters))
    # logger.output_print(args)

    # set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay,
                                 )
    # set up lr_scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=args.lr_drop_size)
    warnings.warn('set T_max in lr_scheduler manually')
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)

    # load checkpoint
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            print('checkpoint: ', args.resume)
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        print('start testing for one epoch !!!')
        with torch.no_grad():
            test_stats = test_one_epoch(model, criterion, data_loader_val, device, logger, args, epoch=0, nprocs=4)
        return
    
    # training
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        # set_trace()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)

        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        test_stats = evaluate(
            model, criterion, data_loader_val, device, logger, args, epoch, nprocs=utils.get_world_size()
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log_train&test.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        if epoch % 5 == 0:
            with torch.no_grad():
                y_true, y_pred = get_predictions(model, dataset_val, device)

            get_multilabel_conf_mat(y_true, y_pred, label_names=args.all_class_name, save_loc=args.output_dir, epoch=epoch)
            cat_metrics = get_metric_per_category(y_true, y_pred, label_names=args.all_class_name, save_loc=args.output_dir, epoch=epoch)
            weighted_metrics = get_weighted_metrics(y_true, y_pred, save_loc=args.output_dir, epoch=epoch)
            
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('OadTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # args.dataset = osp.basename(osp.normpath(args.data_root)).upper()
    with open(args.dataset_file, 'r') as f:
        data_info = json.load(f)['METEOR']
    args.train_session_set = data_info['train_session_set']
    args.test_session_set = data_info['test_session_set']
    args.all_class_name = ["Background", "OverTaking", "LaneChange", "WrongLane", "Cutting"]
    args.numclass = len(args.all_class_name)
    
    args.num_layers = 3
    args.decoder_layers = 4
    args.weight_session_set = 'all'
    args.epochs = 51
    
    args.pickle_file_name = 'features_i3d.pkl'
    args.dim_feature = 2048

    args.output_dir = f'experiments/final/{args.pickle_file_name[:-4]}_old_model'

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
