import random

def encoder_camera_neck_in_channels(depth = None, out_indices = None):
    # depends on encoder_camera_backbone_depth and encoder_camera_backbone_out_indices
    if depth is None:
        raise ValueError("Neck in channels requires encoder.camera.backbone.depth")
    if out_indices is None:
        raise ValueError("Neck in channels requires encoder.camera.backbone.out_indices")
    in_channels = {
        18: [64, 128, 256, 512],
        34: [64, 128, 256, 512],
        50: [256, 512, 1024, 2048],
        101: [256, 512, 1024, 2048],
        152: [256, 512, 1024, 2048],
    }
    return [in_channels[depth][i] for i in out_indices]
