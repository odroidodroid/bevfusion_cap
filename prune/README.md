
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

### 2. Train LiDAR-only

2-1. modify config : configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_prune_0.5_0.5.yaml

    max_epochs : 12
    runner : 
	    type : CustomSPSSPruningEpochBasedRunner
	    max_epochs : ${max_epochs}

2-2. train with the config
