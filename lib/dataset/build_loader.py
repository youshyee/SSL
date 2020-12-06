from functools import partial

from mmcv.runner import get_dist_info
from mmcv.parallel import collate
from torch.utils.data import DataLoader
import numpy as np

# https://github.com/pytorch/pytorch/issues/973
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(dataset, sampler, batch_size, workers_per_gpu, dist=True, **kwargs):
    if dist:
        num_workers = workers_per_gpu
    else:
        raise NotImplementedError

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=sampler,
                             num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True,
                             worker_init_fn=worker_init_fn)

    return data_loader
