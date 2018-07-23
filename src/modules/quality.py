import numpy as np
from sklearn.metrics import roc_curve, average_precision_score


class QualityObject:
    def __init__(self, label, prediction):
        self.label = label
        self.prediction = prediction


def compute_quality(ground_truth_rects, pred_rects):
    pred_rects = list(pred_rects)
    quality_objects = []
    missed_rects = []
    for ground_truth_rect in ground_truth_rects:
        distances = [rect.iou(ground_truth_rect) for rect in pred_rects]
        if len(distances) > 0 and np.max(distances) > 0.5:
            nearest_rect_index = int(np.argmax(distances))
            pred_rect = pred_rects[nearest_rect_index]
            quality_objects.append(QualityObject(1.0, pred_rect.score))
            del pred_rects[nearest_rect_index]
        else:
            missed_rects.append(ground_truth_rect)
    for pred_rect in pred_rects:
        quality_objects.append(QualityObject(0.0, pred_rect.score))
    for true_rect in missed_rects:
        quality_objects.append(QualityObject(1.0, 0.0))

    return quality_objects


def build_roc(quality_objects):
    true_labels = []
    predictions = []
    for q in quality_objects:
        true_labels.append(q.label)
        predictions.append(q.prediction)
    fpr, tpr, threshold = roc_curve(true_labels, predictions)


def compute_mAP(quality_objects):
    true_labels = []
    predictions = []
    for q in quality_objects:
        true_labels.append(q.label)
        predictions.append(q.prediction)
    return average_precision_score(true_labels, predictions)
