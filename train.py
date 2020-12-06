from __future__ import division

import argparse

import torch
import torch.distributed as dist
from mmcv import Config

from lib.api import (get_root_logger, init_dist, set_random_seed, train_model)
from lib.dataset import *
from lib.model import *


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--resume_from',
                        help='the git push -u origin mastercheckpoint file to resume from')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--validate', action='store_true', help='validate during training')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='pytorch',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    validate = args.validate
    # cudnn benchmark enable to accelerate training in fixed input size
    if cfg.get('cudnn_benchmark', True):
        torch.backends.cudnn.benchmark = True

    if args.work_dir is not None:
        # override the workdir
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.launcher == 'none':
        distributed = False
        raise NotImplementedError
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    # ! build model here

    model = eval(cfg.model_type)(**cfg.model)

    # ! dataset here

    transform = Transform(cfg.aug.transforms)
    dataset = eval(cfg.dataset_type)(transform=transform, **cfg.dataset)

    if validate:
        if 'val_dataset' in cfg:
            val_dataset = eval(dataset_type)(**cfg.val_dataset)
        else:
            raise NotImplementedError
        dataset = [dataset, val_dataset]

    if cfg.checkpoint_config is not None:
        #  config file content and checkpoints as meta data
        cfg.checkpoint_config.meta = dict(config=cfg.text)
    # cp cfg file to work_dir

    train_model(model, dataset, cfg, validate=validate, logger=logger)


if __name__ == '__main__':
    main()
