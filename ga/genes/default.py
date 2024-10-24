import random

def image_size():
    variables = [
        {
            "ratio_func": lambda scale: [24*scale, 32*scale],
            "scale_list": list(range(5,12))
        },
        {
            "ratio_func": lambda scale: [72*scale, 128*scale],
            "scale_list": list(range(2,6))
        },
        {
            "ratio_func": lambda scale: [32*scale, 88*scale],
            "scale_list": list(range(4,9))
        },
        {
            "ratio_func": lambda scale: [8*scale, 24*scale],
            "scale_list": list(range(15,31))
        },
    ]
    var = random.choice(variables)
    scale = random.choice(var["scale_list"])
    image_size = var["ratio_func"](scale)
    return image_size
