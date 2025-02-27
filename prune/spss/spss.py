# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import auto_fp16
from torch import nn as nn
from functools import partial

# from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
# from mmdet3d.ops import spconv as spconv
import spconv.pytorch as spconv
from spconv.pytorch import SparseSequential
from mmdet.models import BACKBONES
from prune.spss.spconv_utils import replace_feature
from prune.spss.pruning_block import SpatialPrunedSubmConv3d, SpatialPrunedConvDownsample, SparseSequentialBatchdict

@BACKBONES.register_module()
class VoxelPruningSparseEncoder(nn.Module):
    r"""Sparse encoder for SECOND and Part-A2.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str], optional): Order of conv module.
            Defaults to ('conv', 'norm', 'act').
        norm_cfg (dict, optional): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int, optional): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int, optional): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]], optional):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]], optional):
            Paddings of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        block_type (str, optional): Type of the block to use.
            Defaults to 'conv_module'.
    """

    def __init__(
        self,
        in_channels,
        sparse_shape,
        base_channels=16,
        output_channels=128,
        encoder_channels=((16,), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1,), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        downsample_pruning_ratio=[0.5, 0.5, 0.5],
        pruning_ratio=[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        downsampling_pruning_mode='topk',
        pruning_mode='topk',
        pred_mode='perspective_attn',
        init_cfg=None,
        order=None,
        block_type=None
    ):
        super().__init__()
        # assert block_type in ["conv_module", "basicblock", "postact_block", "conv_block"]
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        # Spconv init all weight on its own
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)        

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            self.base_channels,
            3,
            padding=1,
            indice_key='subm1',
            bias=False
        )
        self.relu1 = nn.ReLU()
        self.bn1 = norm_fn(self.base_channels)
        self.conv_input = SparseSequential(self.conv1, self.bn1, self.relu1)

        encoder_out_channels = self.make_encoder_layers(
            self.base_channels, norm_fn, downsample_pruning_ratio, pruning_ratio, downsampling_pruning_mode, pruning_mode, pred_mode
        )

        self.conv2 = spconv.SparseConv3d(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(1, 1, 3),
            stride=(1, 1, 2),
            padding=0,
            indice_key="spconv_down2"
        )
        self.relu2 = nn.ReLU()
        self.bn2 = norm_fn(self.output_channels)
        self.conv_out = SparseSequential(self.conv2, self.bn2, self.relu2)

    @auto_fp16(apply_to=("voxel_features",))
    def forward(self, voxel_features, coors, batch_size, **kwargs):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """
        
        
        coors = coors.int()
        
        input_sp_tensor = spconv.SparseConvTensor(
            voxel_features, coors, self.sparse_shape, batch_size
        )
        x = self.conv_input(input_sp_tensor)

        batch_dict = {'batch_size' : batch_size,
                      'voxel_features' : voxel_features,
                      'voxel_coords' : coors,
                      'loss_reg_voxel_prob' : 0}

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x, batch_dict = encoder_layer(x, batch_dict)
            encode_features.append(x)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, H, W, D = spatial_features.shape
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous()
        spatial_features = spatial_features.view(N, C * D, H, W)

        return spatial_features, batch_dict['loss_reg_voxel_prob']

    def make_encoder_layers(
        self,
        in_channels,
        norm_fn,
        downsample_pruning_ratio, 
        pruning_ratio, 
        downsampling_pruning_mode, 
        pruning_mode,
        pred_mode='perspective_attn'
    ):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str, optional): Type of the block to use.
                Defaults to 'conv_module'.
            conv_cfg (dict, optional): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        # assert block_type in ["conv_module", "basicblock","prune_block"]
        self.encoder_layers = SparseSequentialBatchdict()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # pruning_ratio_i = tuple(pruning_ratio[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if j == len(blocks) - 1 and i != len(self.encoder_channels) - 1:
                    blocks_list.append(
                        PostActBlock(
                            in_channels,
                            out_channels,
                            3,
                            norm_fn=norm_fn,
                            stride=2,
                            padding=padding,
                            indice_key=f"spconv{i + 1}",
                            conv_type="sprs",
                            downsample_pruning_mode=downsampling_pruning_mode,
                            loss_mode=None,
                            pruning_ratio=downsample_pruning_ratio[i],
                            pred_mode=pred_mode
                        )
                    )
                else :
                    blocks_list.append(
                        PruneSparseBasicBlock(
                            out_channels,
                            out_channels,
                            norm_fn=norm_fn,
                            padding=padding,
                            indice_key=f"subm{i + 1}",
                            conv_types="subm",
                            pruning_mode=pruning_mode,
                            pruning_ratio=pruning_ratio[i]
                        )
                    )
                in_channels = out_channels
            stage_name = f"encoder_layer{i + 1}"
            stage_layers = SparseSequentialBatchdict(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels
    

class PostActBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0, pruning_ratio=0.5,
                   conv_type='subm', norm_fn=None, loss_mode=None, downsample_pruning_mode="topk", pred_mode='perspective_attn'):
        super().__init__()
        self.indice_key = indice_key
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if conv_type == 'subm':
            self.conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        elif conv_type == 'subm_1':
            self.conv = spconv.SubMConv3d(in_channels, out_channels, 1, bias=False, indice_key=indice_key)
        elif conv_type == 'spconv':
            self.conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    bias=False, indice_key=indice_key)
        elif conv_type == 'inverseconv':
            self.conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        elif conv_type == "sprs":
            self.conv = SpatialPrunedConvDownsample(in_channels, out_channels, kernel_size, stride=stride, padding=padding, indice_key=indice_key, bias=False, 
                                                    pred_mode=pred_mode, loss_mode=loss_mode, pruning_mode=downsample_pruning_mode)        
        else:
            raise NotImplementedError        

        self.bn1 = norm_fn(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, batch_dict):
        if isinstance(self.conv, (SpatialPrunedConvDownsample,)):
            x, batch_dict = self.conv(x, batch_dict)
        else:
            x = self.conv(x)
            
        x = x.replace_feature(self.bn1(x.features))
        x = x.replace_feature(self.relu(x.features))
        return x, batch_dict
    
class PruneSparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, indice_key=None, stride=1, padding=0, pruning_ratio=0.5,
                   conv_types='subm', norm_fn=None, downsample=None, pruning_mode="topk"):
        super(PruneSparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = False
        self.indice_key = indice_key
        self.inplanes = inplanes
        self.planes = planes
        if conv_types=="subm" :
            self.conv1 = spconv.SubMConv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key)
            self.conv2 = spconv.SubMConv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key)
        elif conv_types=="spss" :            
            self.conv1 = SpatialPrunedSubmConv3d(
                inplanes, planes, 3, norm_fn=norm_fn, padding=padding, bias=bias, indice_key=indice_key+"_1", pruning_ratio=pruning_ratio, pruning_mode=pruning_mode)
            self.conv2 = SpatialPrunedSubmConv3d(
                planes, planes, 3, norm_fn=norm_fn, padding=padding, bias=bias, indice_key=indice_key+"_2", pruning_ratio=pruning_ratio, pruning_mode=pruning_mode)

        self.bn1 = norm_fn(planes)
        self.bn2 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, batch_dict):
        identity = x
        if isinstance(self.conv1, SpatialPrunedSubmConv3d) :
            out, batch_dict = self.conv1(x, batch_dict)
        else :
            out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))
        if isinstance(self.conv2, SpatialPrunedSubmConv3d) :
            out, batch_dict = self.conv2(out, batch_dict)
        else :
            out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out, batch_dict