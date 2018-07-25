import os.path as osp
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
import config


def create_gauss_image(w, h):
    x = np.linspace(-(w-1)//2, (w-1)//2, w)
    y = np.linspace(-(h-1)//2, (h-1)//2, h)
    normalizer = 0.2
    gauss_x = np.exp(-x ** 2 / (w * normalizer)).reshape([1, w])
    gauss_y = np.exp(-y ** 2 / (h * normalizer)).reshape([h, 1])
    gauss = np.dot(gauss_y, gauss_x)
    return gauss


def max_blend(mask, x1, y1, x2, y2, c, part):
    mask[y1:y2, x1:x2, c] = np.maximum(mask[y1:y2, x1:x2, c], part)


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
    def __init__(self):
        self.image_name = None
        self.image_path = None
        self.rects = None
        self.image = None
        self.mask = None

    @staticmethod
    def create(image, rects, image_path=None):
        self = RectsImage()
        self.rects = rects
        self.image = image
        if image_path is not None:
            self.image_path = image_path
            self.image_name = osp.basename(image_path)
        return self

    @staticmethod
    def load_from_file(image_path, rects_path=None):
        self = RectsImage()
        if rects_path is None:
            rects_path = osp.splitext(image_path)[0] + '.xml'
        self.image_name = osp.basename(image_path)
        self.image_path = image_path
        self.rects = self.load_rects_from_xml(rects_path)
        self.image = None
        self.mask = None
        if config.load_all_images_to_ram:
            self.load()
        return self

    def load(self):
        self.image = cv2.imread(self.image_path)
        assert self.image is not None, 'Image {} is None'.format(self.image_path)
        # self.mask = self.draw_mask()

    def release(self):
        if self.image is not None:
            del self.image
            self.image = None
        if self.mask is not None:
            del self.mask
            self.mask = None

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
        if self.image_path is not None:
            ET.SubElement(root, "filename").text = self.image_name
        if self.image_name is not None:
            ET.SubElement(root, "path").text = osp.basename(self.image_path)
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
            ET.SubElement(bnd_box, "xmin").text = str(int(rect.x))
            ET.SubElement(bnd_box, "ymin").text = str(int(rect.y))
            ET.SubElement(bnd_box, "xmax").text = str(int(rect.x + rect.w))
            ET.SubElement(bnd_box, "ymax").text = str(int(rect.y + rect.h))

        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="\t")
        with open(xml_path, "w") as f:
            f.write(xmlstr)

    def draw_mask(self, rect2rect_transofrm=None):
        if rect2rect_transofrm is None:
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
        else:
            mask = np.zeros([rect2rect_transofrm.h, rect2rect_transofrm.w, config.output_channels_count], np.float32)
            for rect in self.rects:
                transformed_rect, inner_area, is_inside, gauss_size = rect2rect_transofrm.get_rect_inner_area(rect)
                if is_inside:
                    gauss = create_gauss_image(gauss_size[0], gauss_size[1])
                    gauss = gauss[inner_area.y:inner_area.y + inner_area.h, inner_area.x:inner_area.x + inner_area.w]
                    x1 = transformed_rect.x
                    y1 = transformed_rect.y
                    x2 = transformed_rect.x + transformed_rect.w
                    y2 = transformed_rect.y + transformed_rect.h
                    max_blend(mask, x1, y1, x2, y2, 0, gauss)
                    if config.output_channels_count > 1:
                        max_blend(mask, x1, y1, x2, y2, 1, gauss * gauss_size[0] / config.mean_rect_size)
                        max_blend(mask, x1, y1, x2, y2, 2, gauss * gauss_size[1] / config.mean_rect_size)

        return mask
