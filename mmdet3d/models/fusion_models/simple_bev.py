from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS
from mmdet3d.datasets import simple_utils as utils

from .base import Base3DFusionModel

__all__ = ["SimpleBEV"]


@FUSIONMODELS.register_module()
class SimpleBEV(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        # aug
        self.rand_flip = False
        scene_centroid = torch.as_tensor([[0.0, 1.0, 0.0]], device='cuda:0', dtype=torch.float)
        Z, Y, X = 180, 1, 180
        self.Z, self.Y, self.X = Z, Y, X
        self.vox_util = utils.vox.Vox_util(
            Z, Y, X,
            scene_centroid=scene_centroid,
            bounds=[-54.0, 54.0, -5.0, 3.0, -54.0, 54.0],
            assert_cube=False)
        if self.vox_util is not None:
            self.xyz_memA = utils.basic.gridcloud3d(1, Z, Y, X, norm=False)
            self.xyz_camA = self.vox_util.Mem2Ref(self.xyz_memA, Z, Y, X, assert_cube=False)
        else:
            self.xyz_camA = None

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    # "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(self, x, pix_T_cams, cam0_T_camXs) -> torch.Tensor:
        # B = batch size, N = number of cameras, C = 3, H = img height, W = img width
        B, N, C, H, W = x.size()            ## [4, 6, 3, 448, 800] // torch.Size([1, 6, 3, 256, 704])
        # reshape tensors
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        x = __p(x)                          ## [B*N, C, H, W] = [24, 3, 448, 800]
        pix_T_cams_ = __p(pix_T_cams)       ## [24, 4, 4]
        cam0_T_camXs_ = __p(cam0_T_camXs)   ## [24, 4, 4]
        camXs_T_cam0_ = utils.geom.safe_inverse(cam0_T_camXs_) ## [24, 4, 4]

        # Normalize
        device = x.device
        x = (x + 0.5 - torch.as_tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float).reshape(1, 3, 1, 1)) /\
            torch.as_tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float).reshape(1, 3, 1, 1)
        if self.rand_flip:
            B0, _, _, _ = x.shape
            self.rgb_flip_index = np.random.choice([0,1], B0).astype(bool)
            x[self.rgb_flip_index] = torch.flip(x[self.rgb_flip_index], [-1])

        # RGB Encoder [BN, C2, H/8, W/8]
        x = self.encoders["camera"]["backbone"](x)  ## // [torch.Size([6, 192, 32, 88]), torch.Size([6, 384, 16, 44]), torch.Size([6, 768, 8, 22])]
        x = self.encoders["camera"]["neck"](x)      ## torch.Size([24, 128, 56, 100]) // [torch.Size([256, 32, 88]), torch.Size([256, 32, 88])]
        if not isinstance(x, torch.Tensor):
            x = x[0]

        if self.rand_flip:
            x[self.rgb_flip_index] = torch.flip(x[self.rgb_flip_index], [-1])
        BN, C, Hf, Wf = x.shape

        # x = x.view(B, int(BN / B), C, Hf, Wf)
        sy = Hf/float(H)
        sx = Wf/float(W)
        Z, Y, X = self.Z, self.Y, self.X # 200, 8, 200 // 1440 1440 40

        # Unproject image feature to 3d grid, torch.Size([24, 128, 200, 8, 200]) -> torch.Size([4, 6, 128, 200, 8, 200]) B N C Z Y X
        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)
        if self.xyz_camA is not None:
            xyz_camA = self.xyz_camA.to(x.device).repeat(B*N,1,1)
        else:
            xyz_camA = None
        feat_mems_ = self.vox_util.unproject_image_to_mem(
            x,
            utils.basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
            camXs_T_cam0_, Z, Y, X,
            xyz_camA=xyz_camA)
        
        feat_mems = __u(feat_mems_) # B, N, C, Z, Y, X

        mask_mems = (torch.abs(feat_mems) > 0).float()
        feat_mem = utils.basic.reduce_masked_mean(feat_mems, mask_mems, dim=1) # B, C, Z, Y, X

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0,1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0,1], B).astype(bool)
            feat_mem[self.bev_flip1_index] = torch.flip(feat_mem[self.bev_flip1_index], [-1])
            feat_mem[self.bev_flip2_index] = torch.flip(feat_mem[self.bev_flip2_index], [-3])

            if rad_occ_mem0 is not None:
                rad_occ_mem0[self.bev_flip1_index] = torch.flip(rad_occ_mem0[self.bev_flip1_index], [-1])
                rad_occ_mem0[self.bev_flip2_index] = torch.flip(rad_occ_mem0[self.bev_flip2_index], [-3])
        
        return feat_mem.squeeze(3)   # torch.Size([2, 128, 180, 1, 180]) // torch.Size([2, 80, 180, 180])

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                B, *_ = img.shape
                rots = kwargs['rots']
                trans = kwargs['trans']
                intrins = kwargs['intrins']

                __p = lambda x: utils.basic.pack_seqdim(x, B)
                __u = lambda x: utils.basic.unpack_seqdim(x, B)
                intrins_ = __p(intrins)
                pix_T_cams_ = utils.geom.merge_intrinsics(*utils.geom.split_intrinsics(intrins_))
                pix_T_cams = __u(pix_T_cams_)

                velo_T_cams = utils.geom.merge_rtlist(rots, trans)
                cam0_T_camXs = utils.geom.get_camM_T_camXs(velo_T_cams, ind=0)
                feature = self.extract_camera_features(
                    img, pix_T_cams, cam0_T_camXs
                ) # torch.Size([2, 128, 180, 180])
                visualize_bev = True
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points) # torch.Size([2, 256, 180, 180])
                visualize_bev = False
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)
            # if visualize_bev:
            #     bs, bev_ch, bev_h, bev_w = feature.shape
            #     bev_feature = feature[0].reshape(bev_ch, bev_h, bev_w).detach().cpu().numpy()
            #     for bch in range(0, bev_ch) :
            #         channel_feature = bev_feature[bch]
            #         import matplotlib.pyplot as plt
            #         plt.imshow(channel_feature, cmap='viridis')
            #         plt.colorbar()
            #         plt.grid(b=None)
            #         plt.savefig(f'runs/{sensor}_channel_{bch}.png')
            #         plt.close()

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs
