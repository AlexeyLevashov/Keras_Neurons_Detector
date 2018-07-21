from modules.dataset import Dataset
import modules.models.vgg
import modules.models.simple_model
from modules.trainer import Trainer


def main():
    dataset = Dataset()
    model = modules.models.simple_model.FCNModel()
    trainer = Trainer()
    trainer.train(model.model, dataset, model.weights_dir)


if __name__ == '__main__':
    main()
