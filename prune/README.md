
## Pruning Reproduce Guide
### 1. Re-generate pkl file
1-1. modify center (mmdet3d/datasets/nuscenes_dataset.py)

	# the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d) # (0.5, 0.5, 0) -> (0.5, 0.5, 0.5)

1-2. run tools/create_data.py

### 2. Train

2-1. if you want to prune camera and lidar both, use command

    torchpack dist-run -np 1 python tools/prune_both.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/prune_both.yaml

2-2. if you want to train camera without pruning and prune lidar, use command

    torchpack dist-run -np 1 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/voxel_prune.yaml
