import json
import os.path as osp
import os
try:
    import setGPU
except Exception as e:
    pass

# Train
mean_rect_size = (24+60)/2.0
batch_shape = [8, 256, 256, 3]
augmentation_scale_range = [0.5, 1.5]
gpu_devices = None

initial_learning_rate = 0.1
epochs_count = 300

show_outputs_update_time = 12

model_name = 'vgg'
load_all_images_to_ram = True
show_outputs_progress = True
one_batch_overfit = False
save_checkpoints = True
load_weights = False
save_model = True
show_stats = False


# Detection
mask_downsample_rate = 4
output_channels_count = 3
nms_iou_threshold = 0.5
heat_map_min_threshold = 0.3
use_patching = True

patch_size = 512
patch_overlap = 64

if osp.exists('config.json'):
    with open('config.json') as f:
        json_data = json.load(f)
        for v in list(globals().keys()):
            if v in json_data:
                globals()[v] = json_data[v]


if gpu_devices is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
