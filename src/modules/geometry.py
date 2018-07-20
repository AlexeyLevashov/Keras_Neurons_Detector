import os.path as osp
import numpy as np
import cv2
import config


def create_gauss_image(w, h):
    x = np.linspace(0, w-1, w) - w/2.0
    y = np.linspace(0, h-1, h) - h/2.0
    normalizer = 2
    gauss_x = np.exp(-x ** 2 / (w * normalizer)).reshape([w, 1])
    gauss_y = np.exp(-y ** 2 / (h * normalizer)).reshape([1, h])
    gauss = np.dot(gauss_x, gauss_y)
    return gauss


def max_blend(mask, x1, y1, x2, y2, c, part):
    mask[y1:y2 + 1, x1:x2 + 1, c] = np.maximum(mask[y1:y2 + 1, x1:x2 + 1, c], part)


class Rect:
    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class RectsImage:
    def __init__(self, image_path, rects_path=None):
        if rects_path is None:
            rects_path = osp.splitext(image_path)[0] + '.txt'
        self.image = cv2.imread(image_path)
        self.rects = self.load_rects(rects_path)
        self.mask = self.draw_mask()

    @staticmethod
    def load_rects(rects_filepath):
        f = open(rects_filepath, 'r')
        rects = []
        rects_count = int(f.readline())
        for j in range(0, rects_count):
            rect_values = str(f.readline()).split(' ')
            rect = Rect(int(rect_values[0]), int(rect_values[1]), int(rect_values[2]), int(rect_values[3]))
            rects.append(rect)

        return rects

    def draw_mask(self):
        mask = np.zeros([self.image.shape[0], self.image.shape[1], 3], np.float32)
        for rect in self.rects:
            x1 = int(np.clip(rect.x, 0, self.image.shape[1] - 1))
            y1 = int(np.clip(rect.y, 0, self.image.shape[0] - 1))
            x2 = int(np.clip(rect.x + rect.w - 1, 0, self.image.shape[1] - 1))
            y2 = int(np.clip(rect.y + rect.h - 1, 0, self.image.shape[0] - 1))
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            gauss = create_gauss_image(w, h)
            max_blend(mask, x1, y1, x2, y2, 0, gauss)
            max_blend(mask, x1, y1, x2, y2, 1, gauss * rect.w / config.mean_rect_size)
            max_blend(mask, x1, y1, x2, y2, 2, gauss * rect.h / config.mean_rect_size)

        return mask
