import sys
import yaml


def chromosome_default_resnet() -> dict:
    with open('./ga/results/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def chromosome_to_config_dict(chromosome: dict) -> dict:
    """
    Convert the chromosome to nested dictionary.
    It split keys by '.' and create nested dictionary to fit the config format.
    """
    result = {}
    for key, value in chromosome.items():
        keys = key.split('.')
        d = result
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    return result

def deep_update(original: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and key in original:
            deep_update(original[key], value)
        else:
            original[key] = value
    return original

if __name__ == "__main__":
    if len(sys.argv) == 1:
        config = {'image_size': [128, 352], 'model.encoders.camera.vtransform.in_channels': 256, 'model.encoders.camera.vtransform.out_channels': 80, 'model.encoders.camera.vtransform.xbound': [-45, 45, 0.25], 'model.encoders.camera.vtransform.ybound': [-45, 45, 0.25], 'model.encoders.camera.vtransform.zbound': [-10, 10, 20], 'model.encoders.camera.backbone.depth': 18, 'model.encoders.camera.backbone.out_indices': [1, 2, 3], 'model.encoders.camera.backbone.init_cfg': {'checkpoint': 'https://download.pytorch.org/models/resnet18-f37072fd.pth', 'type': 'Pretrained'}, 'model.encoders.camera.neck.in_channels': [128, 256, 512], 'model.encoders.lidar.backbone.encoder_channels': [[16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128]], 'model.encoders.lidar.backbone.encoder_paddings': [[0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]], 'model.encoders.lidar.backbone.output_channels': 128}
    elif len(sys.argv) == 2:
        config = eval(sys.argv[1])
        if not isinstance(config, dict):
            print("config_str must be a dictionary")
            sys.exit(1)
    else:
        print("Usage: python test_config.py [config_str]")
        sys.exit(1)
    config = chromosome_to_config_dict(config)
    full_config = deep_update(chromosome_default_resnet(), config)
    with open('./test/bevfusion.yaml', 'w') as f:
        yaml.dump(full_config, f)
