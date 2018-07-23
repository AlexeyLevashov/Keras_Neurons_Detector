import keras.backend as K


class LayerStat:
    def __init__(self, tensor_name, tensor_batch):
        self.tensor_name = tensor_name
        self.min_value = tensor_batch.min()
        self.mean_value = tensor_batch.mean()
        self.max_value = tensor_batch.max()
        self.std_value = tensor_batch.std()

    def to_string(self):
        return "{}: {} {} {} (+/- {})".format(self.tensor_name, self.min_value, self.mean_value,
                                              self.max_value, self.std_value)


def get_stats(fcn_model, images_batch):
    keras_call = K.function(fcn_model.model.inputs, fcn_model.tensors)
    tensors_batches = keras_call([images_batch])
    stats = [LayerStat(fcn_model.tensors[i].name, tensor_batch) for i, tensor_batch in enumerate(tensors_batches)]
    return stats
