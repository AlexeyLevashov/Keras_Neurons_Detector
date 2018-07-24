import os.path as osp
import tensorflow as tf
from modules.utils import preprocess_batch
from modules.geometry import Rect
import config


class FCNDetector:
    def __init__(self, model=None, weights_path=None):
        self.model = model
        if weights_path is not None and osp.exists(weights_path):
            self.model.load_weights(weights_path)

        with tf.device('/cpu:0'):
            self.local_cpu_sess = tf.Session()
            self.heat_map_placeholder = tf.placeholder(tf.float32, [None, None, None, None])
            self.max_pooled_heat_map = tf.nn.max_pool(self.heat_map_placeholder, ksize=[1, 3, 3, 1],
                                                      strides=[1, 1, 1, 1],
                                                      padding='SAME')

    def predict_heat_maps_batch(self, images_batch):
        batch_heat_maps = self.model.predict(preprocess_batch(images_batch))
        return batch_heat_maps

    def heat_map_nms(self, heat_map):
        heat_map_init = heat_map[:, :, 0]
        with tf.device('/cpu:0'):
            heat_map_batch = heat_map_init.reshape([1, heat_map_init.shape[0], heat_map_init.shape[1], 1])
            nms_heat_map = self.local_cpu_sess.run(self.max_pooled_heat_map,
                                                   feed_dict={self.heat_map_placeholder: heat_map_batch})
            nms_heat_map = nms_heat_map[0, :, :, 0]
            nms_heat_map = heat_map_init - nms_heat_map
            nms_heat_map = (nms_heat_map > -0.00000001) * (heat_map_init > 0.1)

        return nms_heat_map

    @staticmethod
    def obtain_rects(nms_heat_map, heat_map):
        non_zero_elems = nms_heat_map.nonzero()
        rects = []
        for i in range(len(non_zero_elems[0])):
            x = non_zero_elems[1][i]
            y = non_zero_elems[0][i]
            score = heat_map[y, x, 0]
            w = heat_map[y, x, 1]*config.mean_rect_size
            h = heat_map[y, x, 2]*config.mean_rect_size

            x *= config.mask_downsample_rate
            y *= config.mask_downsample_rate

            x -= w/2
            y -= h/2

            rects.append(Rect(x, y, w, h, score))

        return rects

    @staticmethod
    def rects_nms(rects):
        output_rects = []
        rects = sorted(rects, key=lambda x: -x.score)
        for rect in rects:
            nearest_rects = [1 for out_rect in output_rects
                             if rect.iou(out_rect) > config.nms_iou_threshold]
            if len(nearest_rects) == 0:
                output_rects.append(rect)

        return output_rects
