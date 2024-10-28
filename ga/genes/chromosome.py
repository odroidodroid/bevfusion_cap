import yaml
import types

from .camera import *
from .lidar import *
from .default import *


"""
# Define the search space
The search space is a dictionary where the keys are the chromosome keys and the 
values are the functions that generate the values for the chromosome keys.

It must follow dependencies between the keys, so the functions must be defined
in the correct order.
"""
parameters = {
    "image_size": image_size,
    "model.encoders.camera.vtransform.in_channels": encoder_camera_vtransform_in_channels,
    "model.encoders.camera.vtransform.out_channels": encoder_camera_vtransform_out_channels,
    "model.encoders.camera.vtransform.xbound": encoder_camera_vtransform_xbound,
    "model.encoders.camera.vtransform.ybound": encoder_camera_vtransform_ybound,
    "model.encoders.camera.vtransform.zbound": encoder_camera_vtransform_zbound,
    "model.encoders.camera.backbone.depth": encoder_camera_backbone_depth,
    "model.encoders.camera.backbone.out_indices": encoder_camera_backbone_out_indices,
    "model.encoders.camera.backbone.init_cfg": encoder_camera_backbone_init_cfg,
    "model.encoders.camera.neck.in_channels": encoder_camera_neck_in_channels,
    "model.encoders.lidar.backbone.encoder_channels": encoder_lidar_backbone_encoder_channels,
    "model.encoders.lidar.backbone.encoder_paddings": encoder_lidar_backbone_encoder_paddings,
    "model.encoders.lidar.backbone.output_channels": encoder_lidar_backbone_output_channels,
}

search_space = {
    "image_size": image_size,
    "model.encoders.camera.vtransform.xbound": encoder_camera_vtransform_xbound,
    "model.encoders.camera.backbone.init_cfg": encoder_camera_backbone_init_cfg,
    "model.encoders.camera.neck.in_channels": encoder_camera_neck_in_channels,
    "model.encoders.lidar.backbone.encoder_paddings": encoder_lidar_backbone_encoder_paddings,
}


def resolve_dependencies(key, chromosome) -> list:
    args = []
    if key == "model.encoders.camera.vtransform.ybound":
        xbound = chromosome["model.encoders.camera.vtransform.xbound"]
        args = [xbound]
    elif key == "model.encoders.camera.backbone.init_cfg":
        depth = chromosome["model.encoders.camera.backbone.depth"]
        args = [depth]
    elif key == "model.encoders.camera.neck.in_channels":
        depth = chromosome["model.encoders.camera.backbone.depth"]
        out_indices = chromosome["model.encoders.camera.backbone.out_indices"]
        args = [depth, out_indices]
    elif key == "model.encoders.lidar.backbone.encoder_paddings":
        encoder_channels = chromosome["model.encoders.lidar.backbone.encoder_channels"]
        args = [encoder_channels]
    # TODO: Add your dependencies
    else:
        pass
    return args
    "model.encoders.lidar.backbone.output_channels": encoder_lidar_backbone_output_channels,


def generate_chromosome() -> dict:
    chromosome = {}
    for key, func in parameters.items():
        args = resolve_dependencies(key, chromosome)
        chromosome[key] = func(*args)
    return chromosome

def cross_over(chr1, chr2):
    ind1 = {}
    ind2 = {}
    for key in chr1.keys():
        if key in search_space:
            ind1[key] = chr1[key]
            ind2[key] = chr2[key]

    child1 = {}
    child2 = {}
    keys = list(ind1.keys())
    points = sorted(random.sample(range(1, len(ind1)), 2)
    points = [0] + points + [len(ind1)]
    
    for i in range(len(points)-1):
        if i % 2 == 0:
            for j in range(points[i], points[i+1]):
                child1[keys[j]] = ind1[keys[j]]
                child2[keys[j]] = ind2[keys[j]]
        else:
            for j in range(points[i], points[i+1]):
                child1[keys[j]] = ind2[keys[j]]
                child2[keys[j]] = ind1[keys[j]]
    
    child1.update(parameters)
    child2.update(parameters)
    for key, val in child1.items():
        if isinstance(val, types.Functiontypes):
            args = resolve_dependencies(key, child1)
            child1[key] = val(*args)

    return child1, child2
      
def mutation(chromosome):
    ind = {}
    for key in chromosome.keys():
        if key in search_space:
            ind[key] = chromosome[key]
    
    keys = list(ind.keys())
    point = random.randint(0, len(keys)-1)
    
    chromosome[keys[point]] = generate_chromosome()[keys[point]]
    
    return chromosome   

def chromosome_to_config(chromosome) -> list:
    # deprecated
    return [f"{key}={value}" for key, value in chromosome.items()]


def chromosome_to_config_dict(chromosome: dict) -> dict:
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


def chromosome_default_resnet() -> dict:
    with open('./ga/results/default.yaml', 'r') as f:
        default = yaml.safe_load(f)
    return default

def chromosome_minidataset() -> dict:
    train_pkl = 'mini_infos_train.pkl'
    valid_pkl = 'mini_infos_val.pkl'
    return {
        "samples_per_gpu": 2,
        "workers_per_gpu": 2,
        "train": {
            "type": "CBGSDataset",
            "dataset": {
                "type": "${dataset_type}",
                "dataset_root": "${dataset_root}",
                "ann_file": f"${{dataset_root + \"{train_pkl}\"}}",
                "pipeline": "${train_pipeline}",
                "object_classes": "${object_classes}",
                "map_classes": "${map_classes}",
                "modality": "${input_modality}",
                "test_mode": False,
                "use_valid_flag": True,
                "box_type_3d": "LiDAR"
            }
        },
        "val": {
            "type": "${dataset_type}",
            "dataset_root": "${dataset_root}",
            "ann_file": f"${{dataset_root + \"{valid_pkl}\"}}",
            "pipeline": "${test_pipeline}",
            "object_classes": "${object_classes}",
            "map_classes": "${map_classes}",
            "modality": "${input_modality}",
            "test_mode": False,
            "box_type_3d": "LiDAR"
        },
        "test": {
            "type": "${dataset_type}",
            "dataset_root": "${dataset_root}",
            "ann_file": f"${{dataset_root + \"{valid_pkl}\"}}",
            "pipeline": "${test_pipeline}",
            "object_classes": "${object_classes}",
            "map_classes": "${map_classes}",
            "modality": "${input_modality}",
            "test_mode": True,
            "box_type_3d": "LiDAR"
        }
    }
