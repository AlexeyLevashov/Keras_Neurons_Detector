import os.path as osp
import tensorflow as tf
import numpy as np
from modules.utils import preprocess_batch, patch_covering
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

    def predict_heatmap(self, images_batch):
        images_batch_ = images_batch
        if len(images_batch.shape) == 3:
            images_batch_ = np.asarray([images_batch_])
        batch_heat_maps = self.model.predict(preprocess_batch(images_batch_))
        if len(images_batch.shape) == 3:
            batch_heat_maps = batch_heat_maps[0]
        return batch_heat_maps

    def predict_heatmap_by_patching(self, image):
        patch_size = config.patch_size
        patch_overlap = config.patch_overlap
        heatmap_patch_overlap = config.patch_overlap // config.mask_downsample_rate
        cc = config.output_channels_count
        downsample_rate = config.mask_downsample_rate
        padding_y = image.shape[0] % config.mask_downsample_rate
        padding_x = image.shape[1] % config.mask_downsample_rate
        image = np.pad(image, ((patch_overlap, patch_overlap + padding_y), (patch_overlap, patch_overlap +padding_x),
                               (0, 0)), 'edge')
        image = np.asarray(image, np.float32)
        image_heatmap = np.zeros([image.shape[0]//downsample_rate, image.shape[1] // downsample_rate, cc], np.float32)
        for range_y in patch_covering(patch_size, patch_overlap*2, image.shape[0]):
            for range_x in patch_covering(patch_size, patch_overlap*2, image.shape[1]):
                heatmap_range_y = np.asarray([(range_y[0] + patch_overlap) // config.mask_downsample_rate,
                                              (range_y[1] - patch_overlap) // config.mask_downsample_rate])
                heatmap_range_x = np.asarray([(range_x[0] + patch_overlap) // config.mask_downsample_rate,
                                              (range_x[1] - patch_overlap) // config.mask_downsample_rate])

                image_part = image[range_y[0]:range_y[1], range_x[0]:range_x[1], :]
                heatmap_part = self.predict_heatmap(image_part)
                image_heatmap[heatmap_range_y[0]:heatmap_range_y[1], heatmap_range_x[0]:heatmap_range_x[1], :] = \
                    heatmap_part[heatmap_patch_overlap:-heatmap_patch_overlap,
                                 heatmap_patch_overlap:-heatmap_patch_overlap]

        image_heatmap = image_heatmap[heatmap_patch_overlap: -heatmap_patch_overlap,
                                      heatmap_patch_overlap: -heatmap_patch_overlap]

        return image_heatmap

    def heat_map_nms(self, heat_map):
        heat_map_init = heat_map[:, :, 0]
        with tf.device('/cpu:0'):
            heat_map_batch = heat_map_init.reshape([1, heat_map_init.shape[0], heat_map_init.shape[1], 1])
            nms_heat_map = self.local_cpu_sess.run(self.max_pooled_heat_map,
                                                   feed_dict={self.heat_map_placeholder: heat_map_batch})
            nms_heat_map = nms_heat_map[0, :, :, 0]
            nms_heat_map = heat_map_init - nms_heat_map
            nms_heat_map = (nms_heat_map > -0.00000001) * (heat_map_init > config.heat_map_min_threshold)

        return nms_heat_map

    @staticmethod
    def obtain_rects(nms_heat_map, heat_map):
        non_zero_elems = nms_heat_map.nonzero()
        rects = []
        for i in range(len(non_zero_elems[0])):
            x = non_zero_elems[1][i]
            y = non_zero_elems[0][i]
            score = heat_map[y, x, 0]
            w = heat_map[y, x, 1]/score*config.mean_rect_size
            h = heat_map[y, x, 2]/score*config.mean_rect_size

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
