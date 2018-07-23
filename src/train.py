from modules.dataset import Dataset
import modules.models.vgg
import modules.models.simple_model
from modules.trainer import Trainer


def main():
    dataset = Dataset()
    # fcn_model = modules.models.simple_model.FCNModel()
    fcn_model = modules.models.vgg.FCNModel()
    trainer = Trainer()
    trainer.train(fcn_model, dataset)


if __name__ == '__main__':
    main()
