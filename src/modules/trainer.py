import numpy as np
import os.path as osp
import os
import tensorflow as tf
import time
import keras.optimizers as optimizers
import keras.backend as K
from keras.callbacks import LambdaCallback, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from modules.images_viewer import ImagesViewer
from modules.utils import preprocess_batch, postprocess_mask
import config


class Trainer:
    def __init__(self):
        self.model = None
        self.dataset = None
        self.writer = None
        self.images_viewer = None
        self.merged = None
        self.last_checked_time = 0
        self.global_step = 0
    
    def generator(self, is_train):
        while 1:
            images, masks = self.dataset.get_batch(is_train=is_train, use_augmentation=is_train)
            yield preprocess_batch(images), masks
    
    def on_batch_end(self, batch, _):
        self.global_step += 1
        current_time = time.time()
    
        if current_time - self.last_checked_time > config.show_outputs_update_time:
            images, masks = self.dataset.get_batch(is_train=1, use_augmentation=1)
            max_index = np.argmax([mask.sum() for mask in masks])
            images = images[max_index:max_index + 1]
            masks = masks[max_index:max_index + 1]
            preprocessed_images = preprocess_batch(images)

            predictions = self.model.predict(preprocessed_images)
    
            if config.show_outputs_progress:
                prediction = postprocess_mask(predictions)[0] * 255
                mask = postprocess_mask(masks)[0] * 255
                self.images_viewer.set_images([images[0], mask, prediction])
    
            self.last_checked_time = current_time

    def train(self, model, dataset, weights_dir):
        def on_batch_end(*args, **kwargs):
            return self.on_batch_end(args, kwargs)

        def generator(*args, **kwargs):
            return self.generator(*args, **kwargs)

        self.model = model
        self.dataset = dataset

        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mean_squared_error', optimizer=sgd, loss_weights=[1])
    
        if config.show_outputs_progress:
            self.images_viewer = ImagesViewer()
            self.images_viewer.start()

        batch_callback = LambdaCallback(on_batch_end=on_batch_end)
        callbacks = []
        if config.show_outputs_progress:
            callbacks.append(batch_callback)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=10, min_lr=0.000001, verbose=1)
        callbacks.append(reduce_lr)

        if not osp.exists(weights_dir):
            os.makedirs(weights_dir)

        save_path1 = osp.join(weights_dir, 'weights.{epoch:02d}-{val_loss:.8f}.hdf5')
        check_pointer1 = ModelCheckpoint(save_path1, save_best_only=True, verbose=1, period=10)
        save_path2 = osp.join(weights_dir, 'best_weights.hdf5')
        check_pointer2 = ModelCheckpoint(save_path2, save_best_only=True, verbose=1)

        if config.save_checkpoints:
            callbacks.append(check_pointer1)

        callbacks.append(check_pointer2)

        logs_dir = osp.join(weights_dir, 'logs')
        if not osp.exists(logs_dir):
            os.makedirs(logs_dir)

        callbacks.append(TensorBoard(logs_dir))

        if osp.exists(save_path2) and config.load_weights:
            self.model.load_weights(save_path2)
    
        self.model.fit_generator(generator(True),
                                 steps_per_epoch=len(self.dataset.train_indices),
                                 epochs=150,
                                 validation_data=generator(False),
                                 validation_steps=len(self.dataset.test_indices),
                                 callbacks=callbacks)
