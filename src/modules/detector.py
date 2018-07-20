import numpy as np
import cv2
import config
import modules.utils as utils
from keras.callbacks import LambdaCallback


def preprocess_batch(batch):
    return (batch - 127.5)/127.5


def postprocess_mask(masks):
    return np.stack([cv2.resize(mask, (0, 0), fx=config.mask_downsample_rate,
                                fy=config.mask_downsample_rate) for mask in masks])


class FCNDetector:
    def __init__(self, model):
        self.model = model
        self.model.compile(loss='mean_squared_error', optimizer='sgd')

    def train(self, dataset):
        def generator(is_train):
            while 1:
                images, masks = dataset.get_batch(is_train=is_train, use_augmentation=is_train)
                yield preprocess_batch(images), masks

        def on_batch_end(batch, _):
            if batch % 4 == 0:
                images, masks = dataset.get_batch(is_train=1, use_augmentation=1)
                image = images[0]
                preprocessed_image = preprocess_batch(image)

                prediction = self.model.predict(np.asarray([preprocessed_image]))
                prediction = postprocess_mask(prediction)[0]*255
                mask = postprocess_mask(masks)[0]*255
                mask[mask > 255] = 255.0
                prediction[prediction > 255] = 255.0
                combined_image = utils.combine_images([image, mask, prediction])
                cv2.imshow('train', combined_image)
                cv2.waitKey(1)

        batch_callback = LambdaCallback(on_batch_end=on_batch_end)

        self.model.fit_generator(generator(True),
                                 steps_per_epoch=len(dataset.train_indices),
                                 epochs=150,
                                 validation_data=generator(False),
                                 validation_steps=len(dataset.test_indices),
                                 callbacks=[batch_callback])
