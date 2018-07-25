import numpy as np
import cv2
import config


def preprocess_batch(batch):
    return (batch - 127.5)/127.5


def postprocess_mask(masks):
    return np.stack([cv2.resize(mask, (0, 0), fx=config.mask_downsample_rate,
                                fy=config.mask_downsample_rate) for mask in masks])


def check_range(r, max_size):
    if r[1] <= max_size:
        return r
    d = r[1] - max_size
    r[1] -= d
    return r


def patch_covering(patch_size, patch_overlap, range_size):
    for x in range(0, range_size, patch_size - patch_overlap):
        range_x = check_range([x, x + patch_size], range_size)
        yield range_x
        if range_x[1] == range_size:
            break
