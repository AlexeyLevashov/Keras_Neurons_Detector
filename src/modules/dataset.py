import os.path as osp
import glob
import numpy as np
import config
import cv2
from modules.geometry import Rect
from modules.transform import Rect2RectTransform
from modules.augmentation import augment
from modules.geometry import RectsImage


class Dataset:
    def __init__(self, labeled_images_dir='../data/labeled_images'):
        train_images_paths = list(glob.glob(osp.join(labeled_images_dir, 'train/*.jpg')))
        test_images_paths = list(glob.glob(osp.join(labeled_images_dir, 'test/*.jpg')))
        self.images_path_list = train_images_paths + test_images_paths
        self.images_data = np.asarray([RectsImage.load_from_file(image_path) for image_path in self.images_path_list])
        indices = list(range(len(self.images_data)))
        self.train_indices = indices[:len(train_images_paths)]
        self.test_indices = indices[len(train_images_paths):]

    def get_batch(self, batch_shape=None, is_train=False, use_augmentation=True):
        indices = self.train_indices if is_train else self.test_indices

        if config.one_batch_overfit:
            np.random.seed(config.one_batch_overfit_seed)
            use_augmentation = False
            indices = self.train_indices

        if batch_shape is None:
            batch_shape = config.batch_shape
        images_batch = []
        masks_batch = []

        augmentation_scale_range = config.augmentation_scale_range
        if not use_augmentation:
            augmentation_scale_range = [1, 1]

        while len(images_batch) < batch_shape[0]:
            index = np.random.randint(0, len(indices))
            image_data = self.images_data[indices[index]]

            if not config.load_all_images_to_ram:
                image_data.load()

            image = image_data.image

            scale_x = np.random.uniform(augmentation_scale_range[0], augmentation_scale_range[1])
            # scale_y = np.random.uniform(augmentation_scale_range[0], augmentation_scale_range[1])
            scale_y = scale_x
            target_h = batch_shape[1]
            target_w = batch_shape[2]
            w = int(target_w * scale_x)
            h = int(target_h * scale_y)
            x = np.random.randint(0, image.shape[1] - w - 1)
            y = np.random.randint(0, image.shape[0] - h - 1)
            image_part = image[y: y + h, x: x + w]
            image_part = cv2.resize(image_part, (target_w, target_h))

            rect2rect_transform = Rect2RectTransform(Rect(x, y, w, h), Rect(0, 0, target_w//config.mask_downsample_rate,
                                                                            target_h//config.mask_downsample_rate))
            mask_part = image_data.draw_mask(rect2rect_transform)
            if use_augmentation:
                image_part, mask_part = augment(image_part, mask_part)

            if not config.load_all_images_to_ram:
                image_data.release()

            if mask_part.sum() < 2.0:
                continue

            images_batch.append(image_part)
            masks_batch.append(mask_part)

        images_batch = np.stack(images_batch)
        masks_batch = np.stack(masks_batch)
        if len(masks_batch.shape) == 3:
            masks_batch = masks_batch.reshape(list(masks_batch.shape) + [1])
        return images_batch, masks_batch
