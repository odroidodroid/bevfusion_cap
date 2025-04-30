import torch
import torch.nn as nn
import torch.nn.functional as F
# import spconv.pytorch as spconv
from spconv.core import ConvAlgo
from torch.nn import Linear, Parameter
from torch.nn.init import constant_, xavier_uniform_

from mmcv.runner import auto_fp16, force_fp32, patch_norm_fp32, wrap_fp16_model
from mmdet3d.ops import spconv
from mmdet3d.ops.spconv import SparseConvTensor
from prune.spss.spconv_utils import replace_feature
from prune.spss.split_voxels import check_repeat, split_voxels_v2


class PointAttention(nn.Module) :
    def __init__(self, embed_dim, dropout=0., bias=True, add_bis_kv=False, add_zero_attn=False, kdim=None, vdim=None) :
        super(PointAttention,self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        
        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias :
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        
        self._reset_parameters()
        
    def _reset_parameters(self) :
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None :
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value) :
        q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, -1)
        return 
    
class SparseSequentialBatchdict(spconv.SparseSequential):
    def __init__(self, *args, **kwargs):
        super(SparseSequentialBatchdict, self).__init__(*args, **kwargs)

    def forward(self, input, batch_dict=None):
        for k, module in self._modules.items():
            if module is None:
                continue
            input, batch_dict = module(input, batch_dict)
        return input, batch_dict


class SpatialPrunedSubmConvBlock(spconv.SparseModule):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 voxel_stride,
                 indice_key=None, 
                 stride=1, 
                 padding=0, 
                 bias=False, 
                 pruning_ratio=0.5,
                 pred_mode="attn_pred",
                 pred_kernel_size=None,
                 point_cloud_range=[-3, -40, 0, 1, 40, 70.4],
                 voxel_size = [0.1, 0.05, 0.05],
                 algo=ConvAlgo.Native,
                 pruning_mode="topk"):
        super().__init__()
        self.indice_key = indice_key
        self.pred_mode =  pred_mode
        self.pred_kernel_size = pred_kernel_size
        self.ori_pruning_ratio= pruning_ratio
        self.pruning_ratio = pruning_ratio
        self.kernel_size = kernel_size
        self.inv_idx =  torch.Tensor([2, 1, 0]).long().cuda()
        
        self.pruning_mode = pruning_mode
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.voxel_stride = voxel_stride
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()
    
        if pred_mode=="learnable":
            assert pred_kernel_size is not None
            self.pred_conv = spconv.SubMConv3d(
                    in_channels,
                    1,
                    kernel_size=pred_kernel_size,
                    stride=1,
                    padding=padding,
                    bias=False,
                    indice_key=indice_key + "_pred_conv",
                    # algo=algo
                )

        self.conv_block = spconv.SubMConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        bias=bias,
                                        indice_key=indice_key,
                                        # subm_torch=False,
                                        # algo=algo
                                    )
        
        self.sigmoid = nn.Sigmoid()
        self.self_attn = PointAttention()
        
    def _combine_feature(self, x_im, x_nim, mask_position):
        assert x_im.features.shape[0] == x_nim.features.shape[0] == mask_position.shape[0]
        new_features = x_im.features
        new_features[mask_position] = x_nim.features[mask_position]
        x_im = replace_feature(x_im, new_features)
        return x_im 

    def get_importance_mask(self, x, voxel_importance):
        batch_size = x.batch_size
        mask_position = torch.zeros(x.features.shape[0],).cuda()
        index = x.indices[:, 0]
        for b in range(batch_size):
            batch_index = index==b
            batch_voxel_importance = voxel_importance[batch_index]
            batch_mask_position = mask_position[batch_index]
            if self.pruning_mode == "topk":
                batch_mask_position_idx = torch.argsort(batch_voxel_importance.view(-1,))[:int(batch_voxel_importance.shape[0]*self.pruning_ratio)]
                batch_mask_position[batch_mask_position_idx] = 1
                mask_position[batch_index] =  batch_mask_position
            elif self.pruning_mode == "thre":
                batch_mask_position_idx = (batch_voxel_importance.view(-1,) <= self.pruning_ratio)
                batch_mask_position[batch_mask_position_idx] = 1
                mask_position[batch_index] =  batch_mask_position
        return mask_position.bool()

    @auto_fp16(apply_to=("x",))
    def forward(self, x, batch_dict):
        # pred importance
        if self.pred_mode=="learnable":
            x_ = x
            x_conv_predict = self.pred_conv(x_)
            voxel_importance = self.sigmoid(x_conv_predict.features) # [N, 1]
        elif self.pred_mode=="attn_pred":
            x_features = x.features
            x_attn_predict = torch.abs(x_features).sum(1) / x_features.shape[1]
            voxel_importance = self.sigmoid(x_attn_predict.view(-1, 1)).half()
        # elif self.pred_mode=='self_attn' :
        #     x_features = x.features
        #     x_self_attn = self_attn(x_features)
        #     voxel_importance = self.sigmoid(x_self_attn)
        else:
             raise Exception('pred_mode is not defined')

        # get mask
        mask_position = self.get_importance_mask(x, voxel_importance)

        # conv
        x_nim = x[~mask_position].half()
        x_im = self.conv_block(x[mask_position])
        # x = x.replace_feature(x.features * voxel_importance)
        # x_nim = x
        # x_im = self.conv_block(x)        

        # mask feature
        out = self._combine_feature(x_im, x_nim, mask_position)
        # out = x_im # ??
        # out = x_nim + x_im # ??
        return out, batch_dict



class SpatialPrunedConvDownsample(spconv.SparseModule):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 indice_key=None, 
                 stride=1, 
                 padding=0, 
                 bias=False, 
                 pruning_ratio=0.5,
                 dilation=1,
                 voxel_stride=1,
                 point_cloud_range=[-3, -40, 0, 1, 40, 70.4],
                 voxel_size=[0.1, 0.05, 0.05],
                 pred_mode="attn_pred",
                 pred_kernel_size=None,
                 algo=ConvAlgo.Native,
                 pruning_mode="topk"):
        super().__init__()
        if isinstance(padding, int):
            self.padding = [padding] * 3
        else:
            self.padding = padding
        self.indice_key = indice_key
        self.stride = stride
        self.dilation = dilation
        self.pred_mode =  pred_mode
        self.pred_kernel_size = pred_kernel_size

        self.pruning_ratio = pruning_ratio
        self.origin_pruning_ratio = pruning_ratio
        self.kernel_size = kernel_size
        self.inv_idx =  torch.Tensor([2, 1, 0]).long().cuda()
        
        self.pruning_mode = pruning_mode
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.voxel_stride = voxel_stride
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()
    
        if pred_mode=="learnable":
            assert pred_kernel_size is not None
            self.pred_conv = spconv.SubMConv3d(
                    in_channels,
                    1,
                    kernel_size=pred_kernel_size,
                    stride=1,
                    padding=padding,
                    bias=False,
                    indice_key=indice_key + "_pred_conv",
                    # algo=algo
                )


        # self.conv_block = spconv.SubMConv3d(
        #                                 in_channels,
        #                                 out_channels,
        #                                 kernel_size=kernel_size,
        #                                 stride=stride,
        #                                 ## stride=1,
        #                                 padding=padding,
        #                                 bias=bias,
        #                                 indice_key=indice_key,
        #                                 # subm_torch=False,
        #                                 # algo=algo
        #                             )

        self.conv_block = spconv.SparseConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        # stride=stride,
                                        stride=1,
                                        padding=padding,
                                        bias=bias,
                                        indice_key=indice_key,
                                        # subm_torch=False,
                                        # algo=algo
                                    )


        _step = int(kernel_size//2)
        kernel_offsets = [[i, j, k] for i in range(-_step, _step+1) for j in range(-_step, _step+1) for k in range(-_step, _step+1)]
        kernel_offsets.remove([0, 0, 0])
        self.kernel_offsets = torch.Tensor(kernel_offsets).cuda().int()  

        self.sigmoid = nn.Sigmoid()

    def gemerate_sparse_tensor(self, x, voxel_importance):
        batch_size = x.batch_size
        voxel_features_im = []    
        voxel_indices_im = []
        voxel_features_nim = []    
        voxel_indices_nim = []
        for b in range(batch_size):
            features_im, indices_im, features_nim, indices_nim = split_voxels_v2(x, b, voxel_importance, self.kernel_offsets, pruning_mode=self.pruning_mode, pruning_ratio=self.pruning_ratio)
            voxel_features_im.append(features_im)
            voxel_indices_im.append(indices_im)
            voxel_features_nim.append(features_nim)
            voxel_indices_nim.append(indices_nim)

        voxel_features_im = torch.cat(voxel_features_im, dim=0)
        voxel_indices_im = torch.cat(voxel_indices_im, dim=0)
        voxel_features_nim = torch.cat(voxel_features_nim, dim=0)
        voxel_indices_nim = torch.cat(voxel_indices_nim, dim=0)
        x_im = SparseConvTensor(voxel_features_im, voxel_indices_im, x.spatial_shape, x.batch_size)
        x_nim = SparseConvTensor(voxel_features_nim, voxel_indices_nim, x.spatial_shape, x.batch_size)
        
        return x_im, x_nim

    def combine_feature(self, x_im, x_nim, remove_repeat=True):
        x_features = torch.cat([x_im.features, x_nim.features], dim=0)
        x_indices = torch.cat([x_im.indices, x_nim.indices], dim=0)
        if remove_repeat:
            index = x_indices[:, 0]
            features_out_list = []
            indices_coords_out_list = []
            for b in range(x_im.batch_size):
                batch_index = index==b
                features_out, indices_coords_out, _ = check_repeat(x_features[batch_index], x_indices[batch_index], flip_first=False)
                features_out_list.append(features_out)
                indices_coords_out_list.append(indices_coords_out)
            x_features = torch.cat(features_out_list, dim=0)
            x_indices = torch.cat(indices_coords_out_list, dim=0)
        
        x_im = replace_feature(x_im, x_features)
        x_im.indices = x_indices
        return x_im

    def reset_spatial_shape(self, x, batch_dict, pair_indices=None, value_mask=None):
        indices = x.indices
        features = x.features
        conv_valid_mask = ((indices[:,1:] % 2).sum(1)==0)
        
        pre_spatial_shape = x.spatial_shape
        new_spatial_shape = []
        for i in range(3):
            size = (pre_spatial_shape[i] + 2 * self.padding[i] - self.dilation *
                    (self.kernel_size - 1) - 1) // self.stride + 1
            if self.kernel_size == -1:
                new_spatial_shape.append(1)
            else:
                new_spatial_shape.append(size)
        indices[:,1:] = indices[:,1:] // 2
        coords = indices[:,1:][conv_valid_mask]
        spatial_indices = (coords[:, 0] >0) * (coords[:, 1] >0) * (coords[:, 2] >0)  * \
            (coords[:, 0] < new_spatial_shape[0]) * (coords[:, 1] < new_spatial_shape[1]) * (coords[:, 2] < new_spatial_shape[2])

        x = SparseConvTensor(features[conv_valid_mask][spatial_indices], indices[conv_valid_mask][spatial_indices].contiguous(), new_spatial_shape, x.batch_size)

        return x

    def forward(self, x, batch_dict):

        if self.pred_mode=="learnable":
            x_ = x
            x_conv_predict = self.pred_conv(x_)
            voxel_importance = self.sigmoid(x_conv_predict.features) # [N, 1]
        elif self.pred_mode=="attn_pred":
            x_features = x.features
            x_attn_predict = torch.abs(x_features).sum(1) / x_features.shape[1]
            voxel_importance = self.sigmoid(x_attn_predict.view(-1, 1)).half()
        else:
             raise Exception('pred_mode is not define')

        x_im, x_nim = self.gemerate_sparse_tensor(x, voxel_importance)
        out = self.combine_feature(x_im, x_nim, remove_repeat=True).half()
        value_mask = None
        out = self.conv_block(out) ## out
        pair_indices = None
        out = self.reset_spatial_shape(out, batch_dict, pair_indices, value_mask).half()
        
        return out, batch_dict