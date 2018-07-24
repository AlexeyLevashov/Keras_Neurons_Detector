from modules.dataset import Dataset
import modules.models.loader as loader
from modules.trainer import Trainer


def main():
    dataset = Dataset()
    fcn_model = loader.get_fcn_model_module().FCNModel()
    trainer = Trainer()
    trainer.train(fcn_model, dataset)


if __name__ == '__main__':
    main()
