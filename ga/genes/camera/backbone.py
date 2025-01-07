import random


def encoder_camera_backbone_depth():
    variables = [18, 34, 50, 101, 152]
    # variables = [50]
    return random.choice(variables)

def encoder_camera_backbone_layers():
    layers = []
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2, 7):
                for l in range(2, 5):
                    layers.append([i, j, k, l])
    
    return random.choice(layers)

def encoder_camera_backbone_init_cfg(layers: list = None):
    if layers is None:
        raise ValueError(
            "Backbone init cfg requires encoder.camera.backbone.depth")
    return {
        'checkpoint': f"/root/once-for-all/subnets/subnet_{layers[0]}{layers[1]}{layers[2]}{layers[3]}.pth",
        'type': 'Pretrained',
    }

# def encoder_camera_backbone_init_cfg(depth: int = None):
#     if depth is None:
#         raise ValueError(
#             "Backbone init cfg requires encoder.camera.backbone.depth")
#     return {
#         'checkpoint': {
#             18: 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
#             34: 'https://download.pytorch.org/models/resnet34-b627a593.pth',
#             50: 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
#             101: 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
#             152: 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
#         }[depth],
#         'type': 'Pretrained',
#     }


def encoder_camera_backbone_out_indices():
     variables = [[0, 1, 2], [1, 2, 3]]
     return random.choice(variables)

def encoder_camera_backbone_first_out_indice():
    variables = [0, 1]
    return random.choice(variables)