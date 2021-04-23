import torch
from advent.dataset.sim_runway import SimRunwayDataSet
from torch.utils import data
import argparse
from advent.domain_adaptation.config import cfg, cfg_from_file
import numpy as np
import random

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--cfg', type=str, default=r'C:\semseg\IntraDA\ADVENT\advent\scripts\configs\advent.yml',
                        help='optional config file', )
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")
    parser.add_argument("--tensorboard", action="store_true",
                        help="visualize training loss with tensorboardX.")
    parser.add_argument("--viz-every-iter", type=int, default=None,
                        help="visualize results.")
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    return parser.parse_args()

args = get_arguments()
print('Called with args:')
print(args)

assert args.cfg is not None, 'Missing cfg file'
cfg_from_file(args.cfg)
_init_fn = None
if not args.random_train:
    torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
    torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
    np.random.seed(cfg.TRAIN.RANDOM_SEED)
    random.seed(cfg.TRAIN.RANDOM_SEED)

    def _init_fn(worker_id):
        np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)
simrunway_dataset = SimRunwayDataSet(root=cfg.DATA_DIRECTORY_SOURCE,
                                 list_path=cfg.DATA_LIST_SOURCE,
                                 set=cfg.TRAIN.SET_SOURCE,
                                 max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_SOURCE,
                                 crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
                                 mean=cfg.TRAIN.IMG_MEAN)
dataloader = data.DataLoader(simrunway_dataset,batch_size=cfg.TRAIN.BATCH_SIZE_SOURCE,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=_init_fn)
dataloader_ = enumerate(dataloader)

_,batch = dataloader_.__next__()
images_source, labels, _, _ = batch
labels.long()
n,h,w = labels.size()
x = labels.view(n,h,w,1)
target_mask = (labels >= 0)*(labels!=255) 
print(target_mask.shape)
labels = labels[target_mask]
print(labels)
print(labels.shape)
print(x.shape)
print(target_mask.view(n, h, w, 1).repeat(1, 1, 1, 4).unique())

c = torch.rand([1,3,2,2])
print(c)
y = torch.tensor(1,3,1,1)
print(y)