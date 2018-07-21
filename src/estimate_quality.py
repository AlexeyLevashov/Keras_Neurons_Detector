import os.path as osp
from modules.dataset import Dataset
from modules.detector import FCNDetector
from modules.images_viewer import ImagesViewer
import modules.models.simple_model
import time


def check_nms():
    dataset = Dataset()
    model = modules.models.simple_model.FCNModel()
    detector = FCNDetector(model.model, osp.join(model.weights_dir, 'best_weights.hdf5'))
    for image_data in dataset.images_data:
        t1 = time.time()
        nms_heat_map = detector.heat_map_nms(image_data.mask)
        if 0:
            images_viewer = ImagesViewer()
            images_viewer.set_images([nms_heat_map*255, image_data.mask*255])
            images_viewer.start()
            images_viewer.join()

        rects = detector.obtain_rects(nms_heat_map, image_data.mask)
        t2 = time.time()
        # print(t2 - t1)
        if len(rects) != len(image_data.rects):
            print("image {}: {} {}", image_data.image_name, len(rects), len(image_data.rects))


if __name__ == '__main__':
    # main()
    check_nms()

