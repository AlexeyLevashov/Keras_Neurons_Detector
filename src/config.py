import json
import os.path as osp


mean_rect_size = (24+60)/2.0
train_split_percent = 0.8
batch_shape = [32, 512, 512, 3]


if osp.exists('config.json'):
    with open('config.json') as f:
        json_data = json.load(f)
        for v in globals():
            if v in json_data:
                globals()[v] = json_data[v]
