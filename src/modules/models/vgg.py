from keras.models import Model
from keras.layers import Input, BatchNormalization, Concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
import tensorflow as tf
import keras.backend as K
import config


class FCNModel:
    def __init__(self):
        self.weights_dir = '../data/trained_weights/vgg/'
        self.tensors = []

        inputs = Input(shape=(None, None, config.batch_shape[3]), name="input_batch")
        self.tensors.append(inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        # x = BatchNormalization()(x)
        self.tensors.append(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = BatchNormalization()(x)
        self.tensors.append(x)

        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = BatchNormalization()(x)
        self.tensors.append(x)

        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        # x = BatchNormalization()(x)
        self.tensors.append(x)

        x = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(3, (1, 1), activation='relu', strides=(1, 1), padding='same', name="output_batch")(x)
        self.tensors.append(x)

        model = Model(inputs=inputs, outputs=x)
        model.summary()
        self.model = model
