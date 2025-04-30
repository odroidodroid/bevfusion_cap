import os, sys
import time
import warnings
import torch
from torch import nn
from . import torch_pruning as tp
from functools import partial

def get_pruner(model, dummy, cfgs) :
    
    unwrapped_parameters = ([])
    
    sparsity_learning = False
    data_dependency = False

    if cfgs.method == "random" :
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=cfgs.global_pruning)
    elif cfgs.method == "l1" :
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=cfgs.global_pruning)
    elif cfgs.method == "lamp" :
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=cfgs.global_pruning)
    elif cfgs.method == "slim" :
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=cfgs.reg, global_pruning=cfgs.global_pruning)
    elif cfgs.method == "group_norm" :
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=cfgs.global_pruning)
    elif cfgs.method == "group_greg" :
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=cfgs.reg, delta_reg=cfgs.delta_reg, global_pruning=cfgs.global_pruning)
    elif cfgs.method == "group_sl" :
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=cfgs.reg, global_pruning=cfgs.global_pruning)
    else :
        raise NotImplementedError
    
    cfgs.data_dependency = data_dependency
    cfgs.sparsity_learning = sparsity_learning
    
    ignored_layers = []
    pruning_ratio_dict = {}
    for m in model.modules() :
        if isinstance(m, nn.Linear) : 
            ignored_layers.append(m)
    round_to = None
    pruner = pruner_entry(model,
                    dummy,
                    importance=imp,
                    iterative_steps=100,
                    pruning_ratio=1.0,
                    pruning_ratio_dict=pruning_ratio_dict,
                    max_pruning_ratio=cfgs.max_pruning_ratio,
                    ignored_layers=ignored_layers,
                    round_to=round_to,
                    unwrapped_parameters=unwrapped_parameters)
    return pruner


def prune_to_target_flops(pruner, model, target_flops, dummy) :
    model.eval()
    ori_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=dummy)
    pruned_ops = ori_ops
    while pruned_ops / 1e9 > target_flops :
        pruner.step()
        pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=dummy)
    return pruned_ops

def prune_to_target_ratio(pruner, model, target_ratio, dummy) :
    model.eval()
    ori_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=dummy)
    pruned_ops = ori_ops
    pruned_ratio = 1 - ((pruned_ops / 1e9) / (ori_ops / 1e9))
    while pruned_ratio < target_ratio :
        pruner.step()
        pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=dummy)
        pruned_ratio = 1 - ((pruned_ops / 1e9) / (ori_ops / 1e9))
    return pruned_ops
