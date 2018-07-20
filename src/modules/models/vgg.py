from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
import config


def build_model():
    inputs = Input(shape=(config.batch_shape[0], config.batch_shape[1], config.batch_shape[2]))
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), dilation_rate=2, activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), dilation_rate=2, activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), dilation_rate=2, activation='relu', padding='same')(x)

    x = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2D(3, (1, 1), strides=(1, 1), padding='same')(x)

    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model
