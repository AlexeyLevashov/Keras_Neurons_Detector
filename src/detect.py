import os.path as osp
import glob
import cv2
import sys
from modules.detector import FCNDetector
import modules.models.loader as loader
import config


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('python detect.py image_path')
        sys.exit()
    images_path_mask = sys.argv[1]

    fcn_model_module = loader.get_fcn_model_module()
    fcn_model = fcn_model_module.FCNModel()
    detector = FCNDetector(fcn_model.model, osp.join(fcn_model.weights_dir, 'best_weights.hdf5'))

    for image_path in glob.glob(images_path_mask):
        image = cv2.imread(image_path)
        heatmap = detector.predict_heatmap_by_patching(image)
