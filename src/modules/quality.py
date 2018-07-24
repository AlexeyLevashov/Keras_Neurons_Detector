import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score


class QualityObject:
    def __init__(self, label, prediction):
        self.label = label
        self.prediction = prediction


def f1(precision, recall):
    return 2*(precision*recall)/(precision + recall)


def match_rects(ground_truth_rects, pred_rects):
    pred_rects = list(pred_rects)
    matched_true_rects = []
    matched_pred_rects = []
    missed_rects = []
    false_positives_rects = []
    for ground_truth_rect in ground_truth_rects:
        distances = [rect.iou(ground_truth_rect) for rect in pred_rects]
        if len(distances) > 0 and np.max(distances) > 0.5:
            nearest_rect_index = int(np.argmax(distances))
            pred_rect = pred_rects[nearest_rect_index]
            matched_true_rects.append(ground_truth_rect)
            matched_pred_rects.append(pred_rect)
            del pred_rects[nearest_rect_index]
        else:
            missed_rects.append(ground_truth_rect)
    for pred_rect in pred_rects:
        false_positives_rects.append(pred_rect)

    return matched_true_rects, matched_pred_rects, missed_rects, false_positives_rects


def compute_quality(ground_truth_rects, pred_rects):
    matched_true_rects, matched_pred_rects, missed_rects, false_positives_rects = match_rects(ground_truth_rects,
                                                                                              pred_rects)
    quality_objects = []
    for pred_rect in matched_pred_rects:
        quality_objects.append(QualityObject(1, pred_rect.score))
    for pred_rect in false_positives_rects:
        quality_objects.append(QualityObject(0, pred_rect.score))
    for true_rect in missed_rects:
        quality_objects.append(QualityObject(1, -1.0))

    return quality_objects


def find_optimal_threshold(quality_objects):
    true_labels = []
    predictions = []
    for q in quality_objects:
        true_labels.append(q.label)
        predictions.append(q.prediction)
    precision, recall, thresholds = precision_recall_curve(true_labels, predictions)
    if thresholds[0] < 0.0:
        del precision[0]
        del recall[0]
        del thresholds[0]

    f1_scores = [(i, f1(precision[i], recall[i])) for i, th in enumerate(thresholds)]
    f1_scores = sorted(f1_scores, key=lambda x: -x[1])
    best_index = f1_scores[0][0]
    best_precision = precision[best_index]
    best_recall = recall[best_index]
    best_f1 = f1(best_precision, best_recall)
    return thresholds[best_index] - 0.00001, best_precision, best_recall, best_f1


def compute_average_precision(quality_objects):
    true_labels = []
    predictions = []
    for q in quality_objects:
        true_labels.append(q.label)
        predictions.append(q.prediction)
    return average_precision_score(true_labels, predictions)
