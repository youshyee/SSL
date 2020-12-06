import os
import os.path as osp

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import Hook
from mmcv.parallel import is_module_wrapper
from torch.utils.data import Dataset
from lib.dataset import build_dataloader

from mmcv.runner import get_dist_info
import tempfile
import shutil
from yxy.debug import dprint


def collect_results(result_part, size, tmpdir=None):
    # result_part should be unordered list
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        # rand = ''.join(map(str, np.random.randint(10, size=3)))
        # tmpdir = tmpdir + '_' + rand
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        total_result_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            total_result_list += mmcv.load(part_file)
        # sort the results
        # ordered_results = []
        # for res in zip(*part_list):
        #     ordered_results.extend(list(res))
        # # the dataloader may pad some samples
        # ordered_results = ordered_results[:size]
        # remove tmp dir

        shutil.rmtree(tmpdir)
        return total_result_list


class DistEvalHook(Hook):
    def __init__(self, dataset, cfg, interval=1):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise TypeError('dataset must be a Dataset object or a dict, not {}'.format(
                type(dataset)))
        self.interval = interval
        self.cfg = cfg

    def after_train_epoch(self, runner):  # fast version of eval
        if not self.every_n_epochs(runner, self.interval):
            return
        print('evaluation')
        dataloader = build_dataloader(dataset=self.dataset,
                                      workers_per_gpu=self.cfg.workers_per_gpu,
                                      batch_size=1,
                                      sampler=torch.utils.data.DistributedSampler(self.dataset),
                                      dist=True)
        if is_module_wrapper(runner.model.module):
            model = runnner.model.module
        else:
            model = runner.model
        model.eval()
        results = []
        rank = runner.rank
        world_size = runner.world_size
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        for i, data in enumerate(dataloader):
            with torch.no_grad():
                result = model.val_step(data, None)
            results.append(result)

            if rank == 0:
                batch_size = 1  # something wrong here
                for _ in range(batch_size * world_size):
                    prog_bar.update()
        model.train()

        # collect results from all ranks
        results = collect_results(results, len(self.dataset),
                                  os.path.join(runner.work_dir, 'temp/cycle_eval'))
        if runner.rank == 0:
            self.evaluate(runner, results)

    def evaluate(self):
        raise NotImplementedError


class DistEvalMeanLossHook(DistEvalHook):
    # results in format of list of dict {total:N,top1:a,top3:b}
    def evaluate(self, runner, results):
        dic = {}
        allsum = 0
        correctsum = 0
        keys = results[0].keys()
        for result in results:
            for key in keys:
                if key not in dic:
                    dic[key] = 0
                dic[key] += result[key]
        for key in dic:
            if key is not "total":
                runner.log_buffer.output['mean' + key] = dic[key] / dic['total']
        runner.log_buffer.ready = True
