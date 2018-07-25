import numpy as np


def augment(image, mask):
    flip_x = np.random.random() < 0.5
    flip_y = np.random.random() < 0.5
    if flip_x:
        image = np.flip(image, 1)
        mask = np.flip(mask, 1)
    if flip_y:
        image = np.flip(image, 0)
        mask = np.flip(mask, 0)

    color_offset = 0.1
    color_mult = 0.1

    image = np.asarray(image)/255
    image += (np.random.random() - 0.5) * color_offset
    image *= (np.random.random() - 0.5) * color_mult + 1.0

    image = np.clip(image*255, 0, 255.0).astype(np.uint8)

    return image, mask
