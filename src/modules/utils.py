import numpy as np
import cv2
import config


def preprocess_batch(batch):
    return (batch - 127.5)/127.5


def postprocess_mask(masks):
    return np.stack([cv2.resize(mask, (0, 0), fx=config.mask_downsample_rate,
                                fy=config.mask_downsample_rate) for mask in masks])
