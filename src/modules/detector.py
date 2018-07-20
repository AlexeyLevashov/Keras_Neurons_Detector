import modules.models.vgg as vgg


def preprocess_batch(batch):
    return (batch - 127.5)/127.5


class NeuronsFCN:
    def __init__(self):
        self.model = vgg.build_model()
        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['loss'])

    def load_weights(self, weights_path='../data/trained_weights/vgg.h5'):
        self.model.load_weights(weights_path)

    def train(self, dataset):
        def train_generator():
            images, masks = dataset.get_batch(is_train=True)
            return preprocess_batch(images), masks

        def test_generator():
            images, masks = dataset.get_batch(is_train=False, use_augmentation=False)
            return preprocess_batch(images), masks

        self.model.fit_generator(train_generator, steps_per_epoch=len(dataset.images_data), epochs=50,
                                 validation_data=test_generator)
