import os.path as osp
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
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
    def __init__(self, x=0, y=0, w=0, h=0, score=1.0, label="neuron"):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.score = score
        self.label = label

    def iou(self, other_rect):
        boxA = [self.x, self.y, self.x + self.w, self.y + self.h]
        boxB = [other_rect.x, other_rect.y, other_rect.x + other_rect.w, other_rect.y + other_rect.h]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        dx = xB - xA
        dy = yB - yA
        if dx <= 0 or dy <= 0:
            return -1.0

        interArea = dx * dy

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def draw(self, image, color, th=1):
        cv2.rectangle(image, (int(self.x), int(self.y)), (int(self.x + self.w), int(self.y + self.h)), color, th)


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

    def save_to_xml(self, xml_path):
        root = ET.Element("annotation")
        root.set('verified', 'yes')
        ET.SubElement(root, "folder").text = 'labeled_images'
        ET.SubElement(root, "filename").text = self.image_name
        ET.SubElement(root, "path").text = self.image_path
        ET.SubElement(ET.SubElement(root, "source"), "database").text = 'Unknown'
        image_size = ET.SubElement(root, "size")
        ET.SubElement(image_size, "width").text = str(self.image.shape[1])
        ET.SubElement(image_size, "height").text = str(self.image.shape[0])
        ET.SubElement(image_size, "depth").text = str(self.image.shape[2])
        ET.SubElement(root, "segmented").text = "0"

        for rect in self.rects:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = rect.label
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bnd_box = ET.SubElement(obj, "bndbox")
            ET.SubElement(bnd_box, "xmin").text = str(rect.x)
            ET.SubElement(bnd_box, "ymin").text = str(rect.y)
            ET.SubElement(bnd_box, "xmax").text = str(rect.x + rect.w)
            ET.SubElement(bnd_box, "ymax").text = str(rect.y + rect.h)

        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="\t")
        with open(xml_path, "w") as f:
            f.write(xmlstr)

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
