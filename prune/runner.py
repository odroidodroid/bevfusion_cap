import re
import time
import warnings
from collections import OrderedDict

import torch

import mmcv
import prune.torch_pruning as tp
import wandb
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import EpochBasedRunner, _load_checkpoint, load_state_dict
from mmcv.runner.builder import RUNNERS
from mmcv.runner.hooks import HOOKS, Fp16OptimizerHook
from mmcv.runner.utils import get_host_info
from mmdet3d.ops.spconv import SparseConv3d, SubMConv3d, SparseSequential
from mmdet3d.ops.voxel import Voxelization
from mmdet3d.runner import CustomEpochBasedRunner
from prune.pruner import prune_to_target_flops, prune_to_target_ratio
from prune.spss.pruning_block import (SpatialPrunedConvDownsample,
                                      SpatialPrunedSubmConvBlock,
                                      SparseSequentialBatchdict)


@RUNNERS.register_module()
class CustomPruningEpochBasedRunner(CustomEpochBasedRunner):

    def __init__(self,
                 pruner=None,
                 prune_cfg=None,
                 *args,
                 **kwargs) :
        super().__init__(*args, **kwargs)
        self.pruner = pruner
        self.prune_cfg = prune_cfg
        
    def set_dataset(self, dataset):
        self._dataset = dataset

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train' or mode == 'finetune' :
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:            
            if mode == 'finetune' :
                if self.prune_cfg.freezed_finetune == 'lidar' :
                    if self._epoch < self.prune_cfg.freeze_epoch :
                        freeze(self.model.module.encoders.lidar.backbone)
                    else :
                        unfreeze(self.model.module.encoders.lidar.backbone)
                elif self.prune_cfg.freezed_finetune == 'camera' :
                    if self._epoch < self.prune_cfg.freeze_epoch :
                        freeze(self.model.module.encoders.camera.backbone)
                    else :
                        unfreeze(self.model.module.encoders.camera.backbone)
                elif self.prune_cfg.freezed_finetune == 'alter' :
                    stop_freeze = self.prune_cfg.freeze_epoch * 2
                    if self._epoch < stop_freeze and self._epoch // 2 == 0 :
                        freeze(self.model.module.encoders.camera.backbone)
                    elif self._epoch < stop_freeze and self._epoch // 2 == 1 :
                        freeze(self.model.module.encoders.lidar.backbone)
                    elif self._epoch > stop_freeze :
                        unfreeze(self.model.module.encoders.camera.backbone)
                        unfreeze(self.model.module.encoders.lidar.backbone)
                else :
                    # finetune all blocks simulataneously
                    pass
            
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if (mode == 'train' or mode == 'finetune') and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def train(self, data_loader, **kwargs):
        super().train(data_loader, **kwargs)

    def prune(self, dataloaders, **kwargs) :
        data = next(iter(dataloaders[0]))
        self.logger.info(f"Camera pruning target ratio : {self.prune_cfg.camera.target_ratio}")
        camera_inputs = data['img'].data[0][0].cuda()
        base_ops, base_params = tp.utils.count_ops_and_params(self.model.module.encoders.camera.backbone, camera_inputs)
        
        prune_to_target_ratio(self.pruner, self.model.module.encoders.camera.backbone, self.prune_cfg.camera.target_ratio, camera_inputs)
        pruned_ops, pruned_size = tp.utils.count_ops_and_params(self.model.module.encoders.camera.backbone, camera_inputs)
        
        self.logger.info("="*16)
        self.logger.info("After camera pruning...")
        self.logger.info("Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(base_params / 1e6, pruned_size / 1e6, pruned_size / base_params * 100))
        self.logger.info("Ops: {:.2f} G => {:.2f} G ({:.2f}%, {:.2f}X )".format(base_ops / 1e9, pruned_ops / 1e9, pruned_ops / base_ops * 100, base_ops / pruned_ops))

        self.model.to('cuda')
        outputs = self.model.module.encoders.camera.backbone(camera_inputs)
        self.model.module.re_init(outputs)

        self.logger.info(f"Lidar pruning target ratio : {self.prune_cfg.lidar.pruning_ratio}")
        lidar_inputs = data['points'].data[0][0].cuda()
        voxelize = Voxelization(voxel_size=[0.075, 0.075, 0.2],
                                point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
                                max_num_points=10)
        feats, coords, sizes = voxelize(lidar_inputs)
        batch_size = coords[-1, 0] + 1
        base_ops, base_params = tp.utils.count_ops_and_params(self.model.module.encoders.lidar.backbone, (feats, coords, batch_size))
        # TODO : replace lidar model or wrapping 
        prune_to_spss_sprs(self.model.module.encoders.lidar.backbone, self.prune_cfg.lidar)
        pruned_ops, pruned_size = tp.utils.count_ops_and_params(self.model.module.encoders.lidar.backbone, lidar_inputs)

        self.logger.info("="*16)
        self.logger.info("After lidar pruning...")
        self.logger.info("Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(base_params / 1e6, pruned_size / 1e6, pruned_size / base_params * 100))
        self.logger.info("Ops: {:.2f} G => {:.2f} G ({:.2f}%, {:.2f}X )".format(base_ops / 1e9, pruned_ops / 1e9, pruned_ops / base_ops * 100, base_ops / pruned_ops))

        self.model = MMDistributedDataParallel(self.model.module.cuda(),
                                        device_ids=[torch.cuda.current_device()],
                                        broadcast_buffers=False,
                                        find_unused_parameters=True)
        
        self.logger.info(f"Model:\n{self.model}")
        
    def finetune(self, data_loader, **kwargs):
        super().train(data_loader, **kwargs)        


    def load_checkpoint(self, 
                        checkpoint, 
                        map_location='cpu', 
                        strict=False, 
                        logger=None, 
                        revise_keys=[(r'^module\.', '')]) :
        lid_only_file_name = checkpoint
                    
        lidar_checkpoint = _load_checkpoint(lid_only_file_name, map_location, logger)
        if not isinstance(lidar_checkpoint, dict) :
            raise RuntimeError(f"No state_dict found in lidar_checkpoint file {lid_only_file_name}")
        if 'state_dict' in lidar_checkpoint :
            lid_state_dict = lidar_checkpoint['state_dict']
        else :
            lid_state_dict = lidar_checkpoint

        checkpoint = lid_state_dict

        revise_keys = [('encoders.lidar.backbone.encoder_layers.encoder_layer1.0.conv1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer1.0.conv1.conv_block.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer1.0.conv2.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer1.0.conv2.conv_block.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer1.1.conv1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer1.1.conv1.conv_block.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer1.1.conv2.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer1.1.conv2.conv_block.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer1.2.0.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer1.2.conv.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer1.2.1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer1.2.bn1.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer1.2.1.bias', 'encoders.lidar.backbone.encoder_layers.encoder_layer1.2.bn1.bias'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer1.2.1.running_mean', 'encoders.lidar.backbone.encoder_layers.encoder_layer1.2.bn1.running_mean'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer1.2.1.running_var', 'encoders.lidar.backbone.encoder_layers.encoder_layer1.2.bn1.running_var'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer2.0.conv1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer2.0.conv1.conv_block.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer2.0.conv2.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer2.0.conv2.conv_block.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer2.1.conv1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer2.1.conv1.conv_block.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer2.1.conv2.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer2.1.conv2.conv_block.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer2.2.0.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer2.2.conv.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer2.2.1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer2.2.bn1.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer2.2.1.bias', 'encoders.lidar.backbone.encoder_layers.encoder_layer2.2.bn1.bias'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer2.2.1.running_mean', 'encoders.lidar.backbone.encoder_layers.encoder_layer2.2.bn1.running_mean'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer2.2.1.running_var', 'encoders.lidar.backbone.encoder_layers.encoder_layer2.2.bn1.running_var'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer3.0.conv1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer3.0.conv1.conv_block.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer3.0.conv2.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer3.0.conv2.conv_block.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer3.1.conv1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer3.1.conv1.conv_block.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer3.1.conv2.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer3.1.conv2.conv_block.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer3.2.0.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer3.2.conv.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer3.2.1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer3.2.bn1.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer3.2.1.bias', 'encoders.lidar.backbone.encoder_layers.encoder_layer3.2.bn1.bias'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer3.2.1.running_mean', 'encoders.lidar.backbone.encoder_layers.encoder_layer3.2.bn1.running_mean'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer3.2.1.running_var', 'encoders.lidar.backbone.encoder_layers.encoder_layer3.2.bn1.running_var'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer4.0.conv1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer4.0.conv1.conv_block.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer4.0.conv2.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer4.0.conv2.conv_block.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer4.1.conv1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer4.1.conv1.conv_block.weight'),
                       ('encoders.lidar.backbone.encoder_layers.encoder_layer4.1.conv2.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer4.1.conv2.conv_block.weight')]
        metadata = getattr(lid_state_dict, '_metadata', OrderedDict())
        lid_state_dict_ = OrderedDict({})
        for k, v in lid_state_dict.items() :
            for p, r in revise_keys:
                if k == p : 
                    lid_state_dict_[r] = v
                else :
                    lid_state_dict_[k] = v
                # if p == k :
                #     if v.ndim > 1 :
                #         v = v.permute(4, 0, 1, 2, 3)
                #     lid_state_dict_[r] = v
                # elif k not in p :
                #     lid_state_dict_[k] = v
        # Keep metadata in state_dict
        lid_state_dict_._metadata = metadata
        # load state_dict
        load_state_dict(self.model.module, lid_state_dict_, strict, logger)

        return checkpoint

def freeze(model) :
    for param in model.parameters() :
        param.requires_grad = False
def unfreeze(model) :
    for param in model.parameters() :
        param.requires_grad = True
    
@HOOKS.register_module()
class Fp16PruningOptimizerHook(Fp16OptimizerHook):
    def after_train_iter(self, runner):
        super().after_train_iter(runner)
        # runner.pruner.step()
        
def prune_to_spss_sprs(model, prune_cfg) :
    ## TODO : only for encoder layers
    for name, module in model.named_modules() :
        if isinstance(module, SparseConv3d) :
            setattr(model, name, WrapperSpatialPrunedConvDownsample(module, prune_cfg))
        elif isinstance(module, SubMConv3d) :
            setattr(module, name, WrapperSpatialPrunedSubmConvBlock(module, prune_cfg))
        elif isinstance(module, SparseSequential) :
            setattr(module, name, WrapperSparseSequentialBatchdict(module, prune_cfg))
    return model

# @RUNNERS.register_module()
# class CustomSPSSPruningEpochBasedRunner(EpochBasedRunner):

#     def __init__(self,
#                  *args,
#                  **kwargs) :
#         super().__init__(*args, **kwargs)
        
#     def set_dataset(self, dataset):
#         self._dataset = dataset

#     def train(self, data_loader, **kwargs):
#         # update the schedule for data augmentation
#         for dataset in self._dataset:
#             dataset.set_epoch(self.epoch)
#         super().train(data_loader, **kwargs)
        
#     def load_checkpoint(self, 
#                         checkpoint, 
#                         map_location='cpu', 
#                         strict=False, 
#                         logger=None, 
#                         revise_keys=[(r'^module\.', '')]) :
#         lid_only_file_name = checkpoint
                    
#         lidar_checkpoint = _load_checkpoint(lid_only_file_name, map_location, logger)
#         if not isinstance(lidar_checkpoint, dict) :
#             raise RuntimeError(f"No state_dict found in lidar_checkpoint file {lid_only_file_name}")
#         if 'state_dict' in lidar_checkpoint :
#             lid_state_dict = lidar_checkpoint['state_dict']
#         else :
#             lid_state_dict = lidar_checkpoint

#         checkpoint = lid_state_dict

#         revise_keys = [('encoders.lidar.backbone.encoder_layers.encoder_layer1.0.conv1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer1.0.conv1.conv_block.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer1.0.conv2.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer1.0.conv2.conv_block.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer1.1.conv1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer1.1.conv1.conv_block.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer1.1.conv2.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer1.1.conv2.conv_block.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer1.2.0.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer1.2.conv.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer1.2.1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer1.2.bn1.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer1.2.1.bias', 'encoders.lidar.backbone.encoder_layers.encoder_layer1.2.bn1.bias'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer1.2.1.running_mean', 'encoders.lidar.backbone.encoder_layers.encoder_layer1.2.bn1.running_mean'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer1.2.1.running_var', 'encoders.lidar.backbone.encoder_layers.encoder_layer1.2.bn1.running_var'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer2.0.conv1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer2.0.conv1.conv_block.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer2.0.conv2.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer2.0.conv2.conv_block.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer2.1.conv1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer2.1.conv1.conv_block.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer2.1.conv2.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer2.1.conv2.conv_block.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer2.2.0.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer2.2.conv.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer2.2.1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer2.2.bn1.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer2.2.1.bias', 'encoders.lidar.backbone.encoder_layers.encoder_layer2.2.bn1.bias'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer2.2.1.running_mean', 'encoders.lidar.backbone.encoder_layers.encoder_layer2.2.bn1.running_mean'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer2.2.1.running_var', 'encoders.lidar.backbone.encoder_layers.encoder_layer2.2.bn1.running_var'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer3.0.conv1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer3.0.conv1.conv_block.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer3.0.conv2.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer3.0.conv2.conv_block.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer3.1.conv1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer3.1.conv1.conv_block.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer3.1.conv2.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer3.1.conv2.conv_block.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer3.2.0.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer3.2.conv.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer3.2.1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer3.2.bn1.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer3.2.1.bias', 'encoders.lidar.backbone.encoder_layers.encoder_layer3.2.bn1.bias'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer3.2.1.running_mean', 'encoders.lidar.backbone.encoder_layers.encoder_layer3.2.bn1.running_mean'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer3.2.1.running_var', 'encoders.lidar.backbone.encoder_layers.encoder_layer3.2.bn1.running_var'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer4.0.conv1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer4.0.conv1.conv_block.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer4.0.conv2.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer4.0.conv2.conv_block.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer4.1.conv1.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer4.1.conv1.conv_block.weight'),
#                        ('encoders.lidar.backbone.encoder_layers.encoder_layer4.1.conv2.weight', 'encoders.lidar.backbone.encoder_layers.encoder_layer4.1.conv2.conv_block.weight')]
#         metadata = getattr(lid_state_dict, '_metadata', OrderedDict())
#         lid_state_dict_ = OrderedDict({})
#         for k, v in lid_state_dict.items() :
#             for p, r in revise_keys:
#                 if k == p : 
#                     lid_state_dict_[r] = v
#                 else :
#                     lid_state_dict_[k] = v
#                 # if p == k :
#                 #     if v.ndim > 1 :
#                 #         v = v.permute(4, 0, 1, 2, 3)
#                 #     lid_state_dict_[r] = v
#                 # elif k not in p :
#                 #     lid_state_dict_[k] = v
#         # Keep metadata in state_dict
#         lid_state_dict_._metadata = metadata
#         # load state_dict
#         load_state_dict(self.model.module, lid_state_dict_, strict, logger)

#         return checkpoint
    
    # def load_checkpoint(self, 
    #                     checkpoint, 
    #                     map_location='cpu', 
    #                     strict=False, 
    #                     logger=None, 
    #                     revise_keys=[('module\.', '')]) :
    #     if isinstance(checkpoint, dict) :
    #         cam_only_file_name = checkpoint["camera"]
    #         lid_only_file_name = checkpoint["lidar"]
            
    #         camera_checkpoint = _load_checkpoint(cam_only_file_name, map_location, logger)
    #         if not isinstance(camera_checkpoint, dict) :
    #             raise RuntimeError(f"No state_dict found in camera_checkpoint file {cam_only_file_name}")
    #         if 'state_dict' in camera_checkpoint :
    #             cam_state_dict = camera_checkpoint['state_dict']
    #         else :
    #             cam_state_dict = camera_checkpoint
    #         revise_keys = [('encoders\.camera\.backbone\.', '')]
    #         metadata = getattr(cam_state_dict, '_metadata', OrderedDict())
    #         for p, r in revise_keys:
    #             cam_state_dict = OrderedDict(
    #                 {re.sub(p, r, k): v if 'encoders.camera.backbone' in k else None for k, v in cam_state_dict.items()})
    #         # Keep metadata in state_dict
    #         cam_state_dict._metadata = metadata
    #         # load state_dict
    #         load_state_dict(self.model.module.encoders.camera.backbone, cam_state_dict, strict, logger)
            
    #         lidar_checkpoint = _load_checkpoint(lid_only_file_name, map_location, logger)
    #         if not isinstance(lidar_checkpoint, dict) :
    #             raise RuntimeError(f"No state_dict found in lidar_checkpoint file {lid_only_file_name}")
    #         if 'state_dict' in lidar_checkpoint :
    #             lid_state_dict = lidar_checkpoint['state_dict']
    #         else :
    #             lid_state_dict = lidar_checkpoint

    #         checkpoint = lid_state_dict

    #         revise_keys = [('encoders\.lidar\.backbone\.', '')]
    #         metadata = getattr(lid_state_dict, '_metadata', OrderedDict())
    #         for p, r in revise_keys:
    #             lid_state_dict = OrderedDict(
    #                 {re.sub(p, r, k): v if 'encoders.lidar.backbone' in k else None for k, v in lid_state_dict.items()})
    #         # Keep metadata in state_dict
    #         lid_state_dict._metadata = metadata
    #         # load state_dict
    #         load_state_dict(self.model.module.encoders.lidar.backbone, lid_state_dict, strict, logger)
            
    #         decoder_state_dict = checkpoint
    #         revise_keys = [('decoder\.', '')]
    #         for p, r in revise_keys:
    #             decoder_state_dict = OrderedDict(
    #                 {re.sub(p, r, k): v if k.startswith('decoder') else None for k, v in decoder_state_dict.items()})
    #         # load state_dict
    #         load_state_dict(self.model.module.decoder, decoder_state_dict, strict, logger)

    #         head_state_dict = checkpoint
    #         revise_keys = [('heads\.', '')]
    #         for p, r in revise_keys:
    #             head_state_dict = OrderedDict(
    #                 {re.sub(p, r, k): v if k.startswith('heads') else None for k, v in head_state_dict.items()})
    #         # load state_dict
    #         load_state_dict(self.model.module.heads, head_state_dict, strict, logger)

    #         return checkpoint

    #     else :
    #         return super().load_checkpoint(checkpoint, map_location, strict, logger, revise_keys)
