import json
import os.path as osp
import os
try:
    import setGPU
except Exception as e:
    pass


mean_rect_size = (24+60)/2.0
train_split_percent = 0.8
batch_shape = [32, 256, 256, 3]
augmentation_scale_range = [1, 1]
gpu_devices = None

mask_downsample_rate = 4
output_channels_count = 3

initial_learning_rate = 0.1
epochs_count = 300

show_outputs_progress = True
one_batch_overfit = False
show_outputs_update_time = 12

save_checkpoints = True
load_weights = False
save_model = True


if osp.exists('config.json'):
    with open('config.json') as f:
        json_data = json.load(f)
        for v in list(globals().keys()):
            if v in json_data:
                globals()[v] = json_data[v]


if gpu_devices is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
