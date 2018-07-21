import numpy as np
import os.path as osp
import os
import tensorflow as tf
import cv2
import time
import keras.optimizers as optimizers
import keras.backend as K
from keras.callbacks import LambdaCallback, ReduceLROnPlateau
from modules.images_viewer import ImagesViewer
import config


def preprocess_batch(batch):
    return (batch - 127.5)/127.5


def postprocess_mask(masks):
    return np.stack([cv2.resize(mask, (0, 0), fx=config.mask_downsample_rate,
                                fy=config.mask_downsample_rate) for mask in masks])


class FCNDetector:
    def __init__(self, model):
        self.model = model
        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mean_squared_error', optimizer=sgd, loss_weights=[1])

    def train(self, dataset):
        info = {'last_checked_time': 0, 'global_step': 0}
        if config.show_outputs_progress:
            images_viewer = ImagesViewer()
            images_viewer.start()

        if config.dump_to_tensorboard:
            writer = tf.summary.FileWriter('../logs', K.get_session().graph)
            merged = tf.summary.merge_all()

            if not osp.exists('../logs/'):
                os.makedirs('../logs/')

        def generator(is_train):
            while 1:
                images, masks = dataset.get_batch(is_train=is_train, use_augmentation=is_train)
                yield preprocess_batch(images), masks

        def on_batch_end(batch, _):
            info['global_step'] += 1
            current_time = time.time()
            if current_time - info['last_checked_time'] > config.show_outputs_update_time:
                images, masks = dataset.get_batch(is_train=1, use_augmentation=1)
                max_index = np.argmax([mask.sum() for mask in masks])
                images = images[max_index:max_index + 1]
                masks = masks[max_index:max_index + 1]
                preprocessed_images = preprocess_batch(images)

                if config.dump_to_tensorboard:
                    sess = K.get_session()
                    input_batch_tensor = sess.graph.get_tensor_by_name('input_batch:0')
                    output_batch_tensor = sess.graph.get_tensor_by_name('output_batch/Relu:0')

                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, predictions = sess.run([merged, output_batch_tensor],
                                                    feed_dict={input_batch_tensor: preprocessed_images},
                                                    options=run_options, run_metadata=run_metadata)

                    writer.add_run_metadata(run_metadata, 'step_{}'.format(info['global_step']))
                    writer.add_summary(summary, info['global_step'])
                    writer.flush()
                else:
                    predictions = self.model.predict(preprocessed_images)

                if config.show_outputs_progress:
                    prediction = postprocess_mask(predictions)[0]*255
                    mask = postprocess_mask(masks)[0]*255
                    images_viewer.set_images([images[0], mask, prediction])

                info['last_checked_time'] = current_time

        batch_callback = LambdaCallback(on_batch_end=on_batch_end)
        callbacks = []
        if config.dump_to_tensorboard or config.show_outputs_progress:
            callbacks = [batch_callback]

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=10, min_lr=0.000001, verbose=1)
        callbacks.append(reduce_lr)

        self.model.fit_generator(generator(True),
                                 steps_per_epoch=len(dataset.train_indices),
                                 epochs=150,
                                 validation_data=generator(False),
                                 validation_steps=len(dataset.test_indices),
                                 callbacks=callbacks)
