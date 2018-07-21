from keras.models import Model
from keras.layers import Input, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
import tensorflow as tf
import keras.backend as K
import config


class FCNModel:
    def __init__(self, build_summary=True):
        self.weights_path = '../data/trained_weights/vgg.h5'

        inputs = Input(shape=(config.batch_shape[1], config.batch_shape[2], config.batch_shape[3]), name="input_batch")
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2D(1, (1, 1), activation='relu', strides=(1, 1), padding='same', name="output_batch")(x)

        if build_summary:
            tf.summary.histogram('conv1d_activations', K.get_session().graph.get_tensor_by_name('conv2d_1/Relu:0'))
            tf.summary.histogram('max_pooling2d_1_activations', K.get_session().graph.get_tensor_by_name('max_pooling2d_1/MaxPool:0'))
            tf.summary.histogram('max_pooling2d_2_activations', K.get_session().graph.get_tensor_by_name('max_pooling2d_2/MaxPool:0'))
            tf.summary.histogram('max_pooling2d_3_activations', K.get_session().graph.get_tensor_by_name('max_pooling2d_3/MaxPool:0'))
            tf.summary.histogram('conv2d_13_activations', K.get_session().graph.get_tensor_by_name('conv2d_13/Relu:0'))
            tf.summary.histogram('conv2d_transpose_1_activations', K.get_session().graph.get_tensor_by_name('conv2d_transpose_1/Relu:0'))
            tf.summary.histogram('output_activations', K.get_session().graph.get_tensor_by_name('output_batch/Relu:0'))

        model = Model(inputs=inputs, outputs=x)
        model.summary()
        self.model = model
