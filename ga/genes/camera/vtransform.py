import copy
import random


def encoder_camera_vtransform_in_channels():
    variables = [256]
    return random.choice(variables)


def encoder_camera_vtransform_out_channels():
    variables = [80]
    return random.choice(variables)


def encoder_camera_vtransform_xbound():
    variables = [
        {
            "bound": lambda interval: 180 * interval,
            "interval": [0.2, 0.25, 0.3, 0.35, 0.4]
        }
    ]
    var = random.choice(variables)
    interval = random.choice(var["interval"])
    bound = var["bound"](interval)
    return [-bound, bound, interval]


def encoder_camera_vtransform_ybound(xbound):
    # same as encoder_camera_vtransform_xbound
    return copy.deepcopy(xbound)


def encoder_camera_vtransform_zbound():
    variables = [
        {
            "bound": lambda interval: interval // 2,
            "interval": [20]
        }
    ]
    var = random.choice(variables)
    interval = random.choice(var["interval"])
    bound = var["bound"](interval)
    return [-bound, bound, interval]
