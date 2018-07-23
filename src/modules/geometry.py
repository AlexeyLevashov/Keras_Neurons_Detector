import os.path as osp
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import config


def create_gauss_image(w, h):
    x = np.linspace(0, w-1, w) - w/2.0
    y = np.linspace(0, h-1, h) - h/2.0
    normalizer = 2
    gauss_x = np.exp(-x ** 2 / (w * normalizer)).reshape([1, w])
    gauss_y = np.exp(-y ** 2 / (h * normalizer)).reshape([h, 1])
    gauss = np.dot(gauss_y, gauss_x)
    return gauss


def max_blend(mask, x1, y1, x2, y2, c, part):
    mask[y1:y2 + 1, x1:x2 + 1, c] = np.maximum(mask[y1:y2 + 1, x1:x2 + 1, c], part)


class Rect:
    def __init__(self, x=0, y=0, w=0, h=0, score=0.0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.score = score


class RectsImage:
    def __init__(self, image_path, rects_path=None):
        if rects_path is None:
            rects_path = osp.splitext(image_path)[0] + '.xml'
        self.image_name = osp.basename(image_path)
        self.image_path = image_path
        self.rects = self.load_rects_from_xml(rects_path)
        self.image = None
        self.mask = None
        if config.load_all_images_to_ram:
            self.load()

    def load(self):
        self.image = cv2.imread(self.image_path)
        self.mask = self.draw_mask()

    def release(self):
        if self.image is not None:
            del self.image
            self.image = None
        if self.mask is not None:
            del self.mask
            self.mask = None

    @staticmethod
    def load_rects_from_txt(rects_filepath):
        f = open(rects_filepath, 'r')
        rects = []
        rects_count = int(f.readline())
        for j in range(0, rects_count):
            rect_values = str(f.readline()).split(' ')
            rect = Rect(int(rect_values[0]), int(rect_values[1]), int(rect_values[2]), int(rect_values[3]))
            rects.append(rect)

        return rects

    @staticmethod
    def load_rects_from_xml(rects_filepath):
        tree = ET.parse(rects_filepath)
        root = tree.getroot()
        rects = []
        for child in root.iter('object'):
            rect = Rect()
            bndbox = child.find('bndbox')
            rect.x = int(bndbox.find('xmin').text)
            rect.y = int(bndbox.find('ymin').text)
            rect.w = int(bndbox.find('xmax').text) - rect.x
            rect.h = int(bndbox.find('ymax').text) - rect.y
            rects.append(rect)
        return rects

    def draw_mask(self):
        mask = np.zeros([self.image.shape[0], self.image.shape[1], config.output_channels_count], np.float32)
        for rect in self.rects:
            x1 = int(np.clip(rect.x, 0, self.image.shape[1] - 1))
            y1 = int(np.clip(rect.y, 0, self.image.shape[0] - 1))
            x2 = int(np.clip(rect.x + rect.w - 1, 0, self.image.shape[1] - 1))
            y2 = int(np.clip(rect.y + rect.h - 1, 0, self.image.shape[0] - 1))
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            gauss = create_gauss_image(w, h)
            max_blend(mask, x1, y1, x2, y2, 0, gauss)
            if config.output_channels_count > 1:
                max_blend(mask, x1, y1, x2, y2, 1, gauss * rect.w / config.mean_rect_size)
                max_blend(mask, x1, y1, x2, y2, 2, gauss * rect.h / config.mean_rect_size)

        return mask
