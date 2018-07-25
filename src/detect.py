import os.path as osp
import glob
import cv2
import sys
import time
from modules.detector import FCNDetector
import modules.models.loader as loader
from modules.geometry import RectsImage
import config


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('python detect.py image_mask_path')
        sys.exit()
    images_path_mask = sys.argv[1]

    fcn_model_module = loader.get_fcn_model_module()
    fcn_model = fcn_model_module.FCNModel()
    detector = FCNDetector(fcn_model.model, osp.join(fcn_model.weights_dir, 'best_weights.hdf5'))

    for image_path in glob.glob(images_path_mask):
        print(image_path)
        image = cv2.imread(image_path)
        time_start = time.time()
        if config.use_patching:
            heatmap = detector.predict_heatmap_by_patching(image)
        else:
            heatmap = detector.predict_heatmap(image)

        nms_heat_map = detector.heat_map_nms(heatmap)
        rects = detector.obtain_rects(nms_heat_map, heatmap)
        reduced_rects = FCNDetector.rects_nms(rects)
        recognized_image = RectsImage.create(image, reduced_rects, image_path=image_path)
        recognized_image.save_to_xml(osp.splitext(image_path)[0] + '.xml')
        time_end = time.time()
        print("Image processing {} takes {} sec".format(image_path, time_end - time_start))
