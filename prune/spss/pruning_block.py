import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter
import copy
from mmcv.runner import auto_fp16
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor
from prune.spss.spconv_utils import replace_feature
from prune.spss.split_voxels import check_repeat, split_voxels_v2

def compress_voxels_by_coord(voxel_features, voxel_coords, coord_idx=0) :
    coord = voxel_coords[:,coord_idx]
    unique_coords, inverse_indices = torch.unique(coord, return_inverse=True)
    compressed_features = torch.zeros(len(unique_coords), voxel_features.size(1), 
                                      device=voxel_features.device, dtype=voxel_features.dtype)
    compressed_features.index_add_(0, inverse_indices, voxel_features)
    counts = torch.bincount(inverse_indices)
    compressed_features /= counts.unsqueeze(1).half()
    return compressed_features, inverse_indices

def attention(q, k):
    scale = q.size(-1) ** 0.5  # Scaling factor for stability
    attn_weights = torch.matmul(k, q.T) / scale
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_weights = torch.matmul(attn_weights, q) / scale
    attn_weights = torch.softmax(attn_weights, dim=-1)
    return attn_weights

class SparseSequentialBatchdict(spconv.SparseSequential):
    def __init__(self, *args, **kwargs):
        super(SparseSequentialBatchdict, self).__init__(*args, **kwargs)

    def forward(self, input, batch_dict=None):
        for k, module in self._modules.items():
            if module is None:
                continue
            input, batch_dict = module(input, batch_dict)
        return input, batch_dict


class SpatialPrunedSubmConv3d(spconv.SparseModule):
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
                 pred_mode="perspective_attn",
                 pred_kernel_size=None,
                 point_cloud_range=[-3, -40, 0, 1, 40, 70.4],
                 voxel_size = [0.1, 0.05, 0.05],
                 pruning_mode="topk",
                 ta=0.9):
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
    
        self.conv_block = spconv.SubMConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        bias=bias,
                                        indice_key=indice_key,
                                    )
        self.sigmoid = nn.Sigmoid()
        
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
        else:
             raise Exception('pred_mode is not defined')
        x_im, x_nim = self.gemerate_sparse_tensor(x, voxel_importance)
        out = self.combine_feature(x_im, x_nim, remove_repeat=True).half()
        out = self.conv_block(out) ## out
        out = self.reset_spatial_shape(out)
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
                 loss_mode=None,
                 pred_mode="perspective_attn",
                 pred_kernel_size=None,
                 pruning_mode="topk",
                 ta=0.9):
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
        self.loss_mode = loss_mode
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.voxel_stride = voxel_stride
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()
    
        self.conv_block = spconv.SparseConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=2,
                                        padding=padding,
                                        bias=bias,
                                        indice_key=indice_key,
                                    )
        self.ta = ta


        _step = int(kernel_size//2)
        kernel_offsets = [[i, j, k] for i in range(-_step, _step+1) for j in range(-_step, _step+1) for k in range(-_step, _step+1)]
        kernel_offsets.remove([0, 0, 0])
        self.kernel_offsets = torch.Tensor(kernel_offsets).cuda().int()  

        self.sigmoid = nn.Sigmoid()
        self.classifier = nn.Linear(in_channels, 2)

    def loss_reg(self, prob) :
        N = prob.shape[0]
        loss = (self.ta - (1/N)*torch.sum(prob))**2
        return loss

    def compute_voxel_prob_softmax(self, voxel_features) :
        logits = self.classifier(voxel_features)
        prob = torch.exp(logits[:,1]) / (torch.exp(logits[:,0]) + torch.exp(logits[:,1]))
        return prob

    def compute_voxel_prob_gumbel(self, voxel_features) :
        logits = self.classifier(voxel_features)
        rand_noise0 = -torch.log(-torch.log(torch.rand(logits[:,0].shape, device=logits.device)))
        rand_noise1 = -torch.log(-torch.log(torch.rand(logits[:,0].shape, device=logits.device)))

        prob = torch.exp(logits[:,1]+rand_noise1) / (torch.exp(logits[:,0]+rand_noise0) + torch.exp(logits[:,1]+rand_noise1))

        return prob

    def compute_voxel_prob_pa(self, voxel_features, voxel_coords) :
        plane_yz, inverse_indices_yz = compress_voxels_by_coord(voxel_features, voxel_coords, 1)
        plane_xz, inverse_indices_xz = compress_voxels_by_coord(voxel_features, voxel_coords, 2)
        voxel_attn_weights = attention(plane_yz, plane_xz)
        voxel_attn_weights = voxel_attn_weights[inverse_indices_yz]
        return voxel_attn_weights

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

        if self.pred_mode=="attn_pred":
            x_features = x.features
            x_attn_predict = torch.abs(x_features).sum(1) / x_features.shape[1]
            voxel_importance = self.sigmoid(x_attn_predict.view(-1, 1)).half()
            x_im, x_nim = self.gemerate_sparse_tensor(x, voxel_importance)
            out = self.combine_feature(x_im, x_nim, remove_repeat=True).half()
            out = self.conv_block(out) ## out
            out = self.reset_spatial_shape(out)
        elif self.pred_mode=="softmax":
            x_features = x.features
            voxel_importance = self.compute_voxel_prob_softmax(x_features)
            if self.training :
                loss_pts_softmax = self.loss_reg(voxel_importance)
                batch_dict['loss_reg_voxel_prob'] += loss_pts_softmax
                x.replace_feature(x_features * voxel_importance)
            else :
                voxel_mask = voxel_importance > self.pruning_ratio
                x.replace_feature(x_features[voxel_mask])
            out = self.conv_block(x)
        elif self.pred_mode=="gumbel_softmax":
            x_features = x.features
            voxel_importance = self.compute_voxel_prob_gumbel(x_features)
            if self.training :
                loss_pts_softmax = self.loss_reg(voxel_importance)
                batch_dict['loss_reg_voxel_prob'] += loss_pts_softmax
                x.replace_feature(x_features * voxel_importance)
            else :
                voxel_mask = voxel_importance > self.pruning_ratio
                x.replace_feature(x_features[voxel_mask])
            out = self.conv_block(x)
        elif self.pred_mode=="perspective_attn":
            x_features = x.features
            x_coords = x.indices
            voxel_importance = self.compute_voxel_prob_pa(x_features, x_coords)
            if self.training :
                loss_pts_softmax = self.loss_reg(voxel_importance)
                batch_dict['loss_reg_voxel_prob'] += loss_pts_softmax
                x.replace_feature(x_features * voxel_importance)
            else :
                voxel_mask = voxel_importance > self.pruning_ratio
                x.replace_feature(x_features[voxel_mask])
            out = self.conv_block(x)
        else:
             raise Exception('pred_mode is not define')

        return out, batch_dict

    def calculate_flops(self, x, batch_dict, mask_position):
        # mask_position = mask_position_ori
        if mask_position.dtype == torch.bool:
            mask_position = torch.nonzero(mask_position).view(-1,)
        pair_indices = copy.deepcopy(x.indice_dict[self.indice_key].indice_pairs)
        pair_indices_in = pair_indices[0] # [k**3, N]
        pair_indices_out = pair_indices[1] # [k**3, N]
        mask = torch.isin(pair_indices_out, mask_position)
        # print("before mask:", (pair_indices_out > -1).sum())
        pair_indices_out[mask] = -1
        # print("after mask:", (pair_indices_out > -1).sum())
        cur_flops = 2 * (pair_indices_out > -1).sum() * self.in_channels * self.out_channels - pair_indices_out.shape[1]
        batch_dict["3dbackbone_flops"] += cur_flops

    def write_obj(self, points, colors, out_filename):
        N = points.shape[0]
        fout = open(out_filename, 'w')
        for i in range(N):
            c = colors[i]
            fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
        fout.close()
