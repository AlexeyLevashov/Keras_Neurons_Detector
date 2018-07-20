import os.path as osp
import glob
import numpy as np
import config
import cv2
from modules.augmentation import augment
from modules.geometry import RectsImage


class Dataset:
    def __init__(self, labeled_images_dir='../data/LabeledImages'):
        self.images_path_list = glob.glob(osp.join(labeled_images_dir, '*.jpg'))
        self.images_data = np.asarray([RectsImage(image_path) for image_path in self.images_path_list])
        np.random.seed(10)
        indices = list(range(len(self.images_data)))
        np.random.shuffle(indices)
        train_count = int(len(self.images_data) * config.train_split_percent)
        self.train_indices = indices[:train_count]
        self.test_indices = indices[train_count:]

    def get_batch(self, is_train=False, use_augmentation=True):
        indices = self.train_indices if is_train else self.test_indices
        batch_shape = config.batch_shape
        images_batch = []
        masks_batch = []

        augmentation_scale_range = config.augmentation_scale_range
        if not use_augmentation:
            augmentation_scale_range = [1, 1]
        for i in range(batch_shape[0]):
            index = np.random.randint(0, len(indices))
            image_data = self.images_data[indices[index]]
            image = image_data.image
            mask = image_data.mask

            scale_x = np.random.uniform(augmentation_scale_range[0], augmentation_scale_range[1])
            scale_y = np.random.uniform(augmentation_scale_range[0], augmentation_scale_range[1])
            target_h = batch_shape[1]
            target_w = batch_shape[2]
            w = int(target_w * scale_x)
            h = int(target_h * scale_y)
            x = np.random.randint(0, image.shape[1] - w - 1)
            y = np.random.randint(0, image.shape[0] - h - 1)
            image_part = image[y: y + h, x: x + w]
            mask_part = mask[y: y + h, x: x + w]
            image_part = cv2.resize(image_part, (target_w, target_h))
            mask_part = cv2.resize(mask_part, (target_w//config.mask_downsample_rate,
                                               target_h//config.mask_downsample_rate))
            if use_augmentation:
                image_part, mask_part = augment(image_part, mask_part)

            images_batch.append(image_part)
            masks_batch.append(mask_part)

        return np.stack(images_batch), np.stack(masks_batch)
