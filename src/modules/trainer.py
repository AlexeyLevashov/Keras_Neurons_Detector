import numpy as np
import os.path as osp
import os
import time
import keras.optimizers as optimizers
from keras.callbacks import LambdaCallback, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from modules.images_viewer import ImagesViewer
from modules.utils import preprocess_batch, postprocess_mask
from modules.tensors_stats import get_stats
import config


class Trainer:
    def __init__(self):
        self.fcn_model = None
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
            init_images, init_masks = self.dataset.get_batch(is_train=1, use_augmentation=1)
            max_index = np.argmax([mask.sum() for mask in init_images])
            images = init_images[max_index:max_index + 1]
            masks = init_masks[max_index:max_index + 1]
            preprocessed_images = preprocess_batch(images)

            predictions = self.model.predict(preprocessed_images)
    
            if config.show_outputs_progress:
                prediction = postprocess_mask(predictions)[0] * 255
                mask = postprocess_mask(masks)[0] * 255
                self.images_viewer.set_images([images[0], mask, prediction])

            if config.show_stats:
                stats = get_stats(self.fcn_model, init_images)
                for stat in stats:
                    print(stat.to_string())

            self.last_checked_time = current_time

    def train(self, fcn_model, dataset):
        def on_batch_end(*args, **kwargs):
            return self.on_batch_end(args, kwargs)

        def generator(*args, **kwargs):
            return self.generator(*args, **kwargs)

        weights_dir = fcn_model.weights_dir
        self.fcn_model = fcn_model
        self.model = fcn_model.model
        self.dataset = dataset

        sgd = optimizers.SGD(lr=config.initial_learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mean_squared_error', optimizer=sgd, loss_weights=[1])
    
        if config.show_outputs_progress:
            self.images_viewer = ImagesViewer()
            self.images_viewer.start()

        batch_callback = LambdaCallback(on_batch_end=on_batch_end)
        callbacks = []
        if config.show_outputs_progress:
            callbacks.append(batch_callback)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=5, min_lr=0.000001, verbose=1)
        callbacks.append(reduce_lr)

        if not osp.exists(weights_dir):
            os.makedirs(weights_dir)

        if config.save_model:
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
                                 epochs=config.epochs_count,
                                 validation_data=generator(False),
                                 validation_steps=len(self.dataset.test_indices),
                                 callbacks=callbacks)

        if config.show_outputs_progress:
            self.images_viewer.alive = False
            self.images_viewer.join()
