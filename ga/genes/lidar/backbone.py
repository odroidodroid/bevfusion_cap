import random

def encoder_lidar_backbone_encoder_channels():
    stage1_list = [[16,32], [16, 16, 32]]
    stage2_list = [[32,64], [32, 32, 64]]
    stage3_list = [[64, 128], [64, 64, 128]]
    stage4_list = [[128], [128, 128]]
    return [
        random.choice(stage1_list),
        random.choice(stage2_list),
        random.choice(stage3_list),
        random.choice(stage4_list)
    ]

def encoder_lidar_backbone_encoder_paddings(channels):
    # depends on encoder_lidar_backbone_encoder_channels
    paddings = {
        '[16, 32]': [0, 1],
        '[16, 16, 32]': [0, 0, 1],
        '[32, 64]': [0, 1],
        '[32, 32, 64]': [0, 0, 1],
        '[64, 128]': [0, [1, 1, 0]],
        '[64, 64, 128]': [0, 0, [1, 1, 0]],
        '[128]': [0],
        '[128, 128]': [0, 0],
    }
    return [paddings[str(ch)] for ch in channels]

def encoder_lidar_backbone_output_channels():
    variables = [128]
    return random.choice(variables)
