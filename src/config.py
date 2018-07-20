import json
import os.path as osp
import os
import setGPU


mean_rect_size = (24+60)/2.0
train_split_percent = 0.8
batch_shape = [1, 512, 512, 3]
augmentation_scale_range = [1, 1]
gpu_devices = None

mask_downsample_rate = 4


if osp.exists('config.json'):
    with open('config.json') as f:
        json_data = json.load(f)
        for v in globals():
            if v in json_data:
                globals()[v] = json_data[v]


if gpu_devices is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
