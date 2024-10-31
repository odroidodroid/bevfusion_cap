import os.path as osp
import pickle
import shutil
import tempfile
import time

import torch
import torch.distributed as dist

import mmcv
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmdet.core import encode_mask_results


def single_gpu_prune_dataset(model, data_loader, ascending, prune_ratio):
    model.eval()
    losses = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            loss = model(return_loss=True, rescale=True, **data)
        losses.append({'loss' : loss['loss'],
                       'token' : data['metas'].data[0][0]['token']})

        prog_bar.update()
    if ascending :        
        losses = sorted(losses, key=lambda x : x["loss"], reverse=False)
    else :
        losses = sorted(losses, key=lambda x : x["loss"], reverse=True)

    remained = int(len(losses) * prune_ratio)
    remained_loss = [losses.pop()['token'] for i in range(remained)]

    return remained_loss
