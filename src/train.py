import os.path as osp
import keras.backend as K
from modules.dataset import Dataset
import modules.models.loader as loader
from modules.detector import FCNDetector
from modules.trainer import Trainer
from estimate_quality import estimate_quality
import config


def main():
    dataset = Dataset()
    fcn_model = loader.get_fcn_model_module().FCNModel()
    trainer = Trainer()
    trainer.train(fcn_model, dataset)
    K.clear_session()

    if not config.one_batch_overfit:
        detector = FCNDetector(fcn_model.model)
        detector.weights_path = osp.join(fcn_model.weights_dir, 'best_weights.hdf5')
        estimate_quality(detector, dataset)


if __name__ == '__main__':
    main()
