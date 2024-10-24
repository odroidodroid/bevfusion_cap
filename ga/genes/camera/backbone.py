import random


def encoder_camera_backbone_depth():
    # variables = [18, 34, 50, 101, 152]
    variables = [50]
    return random.choice(variables)


# TODO load pretrained camera backbone
def encoder_camera_backbone_init_cfg(depth: int = None):
    if depth is None:
        raise ValueError(
            "Backbone init cfg requires encoder.camera.backbone.depth")
    return {
        'checkpoint': {
            50: 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
        }[depth],
        'type': 'Pretrained',
    }


def encoder_camera_backbone_out_indices():
    variables = [[0, 1, 2], [1, 2, 3]]
    return random.choice(variables)
