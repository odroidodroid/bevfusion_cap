import random

def image_size():
    output_size = [[48, 56], [48, 64], [40, 64], [40, 72], [40, 80], [32, 80], [32, 88], [32, 96], [24, 96], [24, 104]]
    selected = random.choice(output_size)
    image_size = [x * 8 for x in selected]
    return image_size
