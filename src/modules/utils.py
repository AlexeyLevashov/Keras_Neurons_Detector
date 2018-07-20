import numpy as np


def combine_images(images):
    w = sum([image.shape[1] for image in images])
    h = images[0].shape[0]
    part_w = images[0].shape[1]

    combined_image = np.zeros([h, w, 3], np.uint8)
    for i, image in enumerate(images):
        combined_image[:, i*part_w:(i+1)*part_w, :] = np.asarray(image, np.uint8)

    return combined_image
