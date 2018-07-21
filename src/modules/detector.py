import numpy as np


class FCNDetector:
    def __init__(self, model, weights_path):
        self.model = model
        self.model.load_weights(weights_path)

    def predict_batch(self, images_batch):
        batch_heat_maps = self.model.predict(images_batch)
        for i in range(batch_heat_maps.shape[0]):
            heat_map = batch_heat_maps[i]

        return batch_heat_maps