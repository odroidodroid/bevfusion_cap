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

