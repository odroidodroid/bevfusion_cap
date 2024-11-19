import types

import yaml
from deap import creator

from ga.utils import configs, logger

from .camera import *
from .default import *
from .lidar import *

CHROMOSOME_DEFAULT_RESNET = None  # Dynamic Loading
CHROMOSOME_DEFAULT = None

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
    key: val for key, val in parameters.items() if key in [
        "image_size",
        "model.encoders.camera.vtransform.xbound",
        "model.encoders.camera.backbone.depth",
        "model.encoders.camera.backbone.out_indices",
        "model.encoders.lidar.backbone.encoder_channels",
    ]}

dependent = {key: val for key, val in parameters.items()
             if key not in search_space}


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


def generate(chr_bp: dict) -> dict:
    """Generate a chromosome from the blueprint."""
    chromosome = chr_bp.copy()
    for key, func in chromosome.items():
        if type(func) == types.FunctionType:
            args = resolve_dependencies(key, chromosome)
            chromosome[key] = func(*args)
    return chromosome


def generate_chromosome() -> dict:
    global CHROMOSOME_DEFAULT
    # In first generation, add default chromosome
    if CHROMOSOME_DEFAULT is None:
        resnet = chromosome_default_resnet()
        resnet = config_dict_to_chromosome(resnet)
        CHROMOSOME_DEFAULT = {key: value for key,
                              value in resnet.items() if key in parameters.keys()}
        return CHROMOSOME_DEFAULT
    return generate(parameters)


def chromosome_to_config(chromosome) -> list:
    """Deprecated"""
    return [f"{key}={value}" for key, value in chromosome.items()]


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


def config_dict_to_chromosome(config_dict: dict) -> dict:
    """
    Convert the config dictionary to a flat chromosome.
    It flattens the nested dictionary to fit the chromosome format.
    """
    def flatten(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict) and new_key not in [
                "model.encoders.camera.backbone.init_cfg",
            ]:
                items.extend(flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    return flatten(config_dict)


def chromosome_default_resnet() -> dict:
    global CHROMOSOME_DEFAULT_RESNET
    if CHROMOSOME_DEFAULT_RESNET is None:
        with open('./ga/results/default.yaml', 'r') as f:
            CHROMOSOME_DEFAULT_RESNET = yaml.safe_load(f)
    return CHROMOSOME_DEFAULT_RESNET


def chromosome_minidataset() -> dict:
    return {
        "samples_per_gpu": 2,
        "workers_per_gpu": 2,
        "train": {
            "type": "CBGSDataset",
            "dataset": {
                "type": "${dataset_type}",
                "dataset_root": "${dataset_root}",
                "ann_file": f"${{dataset_root + \"{configs.MINI_TRAIN_PKL}\"}}",
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
            "ann_file": f"${{dataset_root + \"{configs.MINI_VALID_PKL}\"}}",
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
            "ann_file": f"${{dataset_root + \"{configs.MINI_VALID_PKL}\"}}",
            "pipeline": "${test_pipeline}",
            "object_classes": "${object_classes}",
            "map_classes": "${map_classes}",
            "modality": "${input_modality}",
            "test_mode": True,
            "box_type_3d": "LiDAR"
        }
    }


# Crossovers
def crossover_onepoint(chr1: dict, chr2: dict) -> tuple:
    keys = list(search_space.keys())
    # choose crossover point
    point: int = random.choice(range(len(keys)))
    logger.debug(f"========== Crossover one point: {point} ==========")
    logger.debug(f"Before 1: {chr1}")
    logger.debug(f"Before 2: {chr2}")

    # crossover
    for idx, key in enumerate(keys):
        if idx >= point:
            chr1[key], chr2[key] = chr2[key], chr1[key]

    chr1.update(dependent)
    chr2.update(dependent)
    chr1 = generate(chr1)
    chr2 = generate(chr2)

    logger.debug(f"After 1: {chr1}")
    logger.debug(f"After 2: {chr2}")
    logger.debug("===================================================")
    return creator.Individual(chr1), creator.Individual(chr2)


def crossover_twopoint(chr1: dict, chr2: dict) -> tuple:
    keys = list(search_space.keys())
    # choose two crossover points
    point1, point2 = random.sample(range(len(keys)), 2)
    logger.debug(f"============ Crossover two point: {point1}, {point2} ============")
    logger.debug(f"Before 1: {chr1}")
    logger.debug(f"Before 2: {chr2}")

    # crossover
    for idx, key in enumerate(keys):
        if point1 <= idx < point2:
            chr1[key], chr2[key] = chr2[key], chr1[key]

    chr1.update(dependent)
    chr2.update(dependent)
    chr1 = generate(chr1)
    chr2 = generate(chr2)

    logger.debug(f"After 1: {chr1}")
    logger.debug(f"After 2: {chr2}")
    logger.debug("===================================================")
    return creator.Individual(chr1), creator.Individual(chr2)


def crossover_uniform(chr1: dict, chr2: dict) -> tuple:
    keys = list(search_space.keys())
    logger.debug("========== Crossover uniform ==========")
    logger.debug(f"Before 1: {chr1}")
    logger.debug(f"Before 2: {chr2}")

    for key in keys:
        if random.random() < 0.5:
            chr1[key], chr2[key] = chr2[key], chr1[key]
    
    chr1.update(dependent)
    chr2.update(dependent)
    chr1 = generate(chr1)
    chr2 = generate(chr2)

    logger.debug(f"After 1: {chr1}")
    logger.debug(f"After 2: {chr2}")
    logger.debug("===================================================")
    return creator.Individual(chr1), creator.Individual(chr2)


# Mutation
def mutate_onepoint(chromosome: dict) -> tuple:
    keys = list(search_space.keys())
    # choose mutation point
    key: str = random.choice(keys)
    logger.debug(f"============== Mutation one point: {keys.index(key)} ==============")
    logger.debug(f"Before: {chromosome}")

    # mutation
    chromosome[key] = search_space[key]()
    
    chromosome.update(dependent)
    chromosome = generate(chromosome)

    logger.debug(f"After: {chromosome}")
    logger.debug("===================================================")
    return creator.Individual(chromosome),


def mutate_uniform(chromosome: dict) -> tuple:
    keys = list(search_space.keys())
    logger.debug("========== Mutation uniform ==========")
    logger.debug(f"Before: {chromosome}")

    for key in keys:
        if random.random() < 0.5:
            chromosome[key] = search_space[key]()
    
    chromosome.update(dependent)
    chromosome = generate(chromosome)

    logger.debug(f"After: {chromosome}")
    logger.debug("===================================================")
    return creator.Individual(chromosome),
