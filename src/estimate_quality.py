import numpy as np
import os.path as osp
import json
import cv2
import time
import modules.utils
import modules.detector
from modules.geometry import RectsImage
from modules.images_viewer import ImagesViewer
from modules.dataset import Dataset
from modules.detector import FCNDetector
from modules.images_viewer import ImagesViewer
from modules.quality import compute_quality, compute_average_precision, find_optimal_threshold, get_precision_recall_curve
import modules.models.loader as loader
import config


def get_quality_objects(detector, image_data):
    if not config.load_all_images_to_ram:
        image_data.load()

    if config.use_patching:
        mask = detector.predict_heatmap_by_patching(image_data.image)
    else:
        mask = detector.predict_heatmap(image_data.image)
    nms_heat_map = detector.heat_map_nms(mask)
    rects = detector.obtain_rects(nms_heat_map, mask)
    reduced_rects = FCNDetector.rects_nms(rects)
    quality_objects = compute_quality(image_data.rects, reduced_rects)
    ap_rate = compute_average_precision(quality_objects)

    print("{}:\t ap {}".format(image_data.image_name, ap_rate))

    if not config.load_all_images_to_ram:
        image_data.release()

    return quality_objects


def estimate_quality(detector, dataset):
    train_quality_objects = []
    print("Train")
    for i in dataset.train_indices:
        image_data = dataset.images_data[i]
        train_quality_objects.extend(get_quality_objects(detector, image_data))

    print("Test")
    test_quality_objects = []
    for i in dataset.test_indices:
        image_data = dataset.images_data[i]
        test_quality_objects.extend(get_quality_objects(detector, image_data))

    train_ap_rate = compute_average_precision(train_quality_objects)
    print("Train AP: {}".format(train_ap_rate))
    test_ap_rate = compute_average_precision(test_quality_objects)
    print("Test AP: {}".format(test_ap_rate))

    train_threshold, train_best_precision, train_best_recall, train_best_f1 = \
        find_optimal_threshold(train_quality_objects)
    print("Train F1 score {}, precision {}, recall {}".format(train_best_f1, train_best_precision, train_best_recall))

    test_threshold, test_best_precision, test_best_recall, test_best_f1 = \
        find_optimal_threshold(test_quality_objects)
    print("Test F1 score {}, precision {}, recall {}".format(test_best_f1, test_best_precision, test_best_recall))

    quality_report = dict()
    quality_report['Train'] = {'F1': train_best_f1,
                               'Precision': train_best_precision,
                               'Recall': train_best_recall,
                               'Threshold': train_threshold}

    quality_report['Test'] = {'F1': test_best_f1,
                              'Precision': test_best_precision,
                              'Recall': test_best_recall,
                              'Threshold': test_threshold}

    with open(osp.splitext(detector.weights_path)[0] + '_quality_report.txt', 'w') as f:
        json.dump(quality_report, f)


def main():
    dataset = Dataset()
    fcn_model_module = loader.get_fcn_model_module()
    fcn_model = fcn_model_module.FCNModel()
    detector = FCNDetector(fcn_model.model, osp.join(fcn_model.weights_dir, 'best_weights.hdf5'))
    estimate_quality(detector, dataset)


if __name__ == '__main__':
    main()
