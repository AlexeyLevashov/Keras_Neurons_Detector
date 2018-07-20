import os.path as osp
import glob
import numpy as np
import config
from modules.geometry import RectsImage


class Dataset:
    def __init__(self, labeled_images_dir='../data/LabeledImages'):
        self.images_path_list = glob.glob(osp.join(labeled_images_dir, '*.jpg'))
        self.images = np.asarray([RectsImage(image_path) for image_path in self.images_path_list])
        np.random.seed(10)
        indices = list(range(len(self.images)))
        np.random.shuffle(indices)
        train_count = int(len(self.images)*config.train_split_percent)
        self.train_indices = indices[:train_count]
        self.test_indices = indices[train_count:]
