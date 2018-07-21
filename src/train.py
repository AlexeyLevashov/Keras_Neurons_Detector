from modules.reader import Dataset
from modules.detector import FCNDetector
import modules.models.vgg
import modules.models.simple_model


def main():
    dataset = Dataset()
    vgg = modules.models.simple_model.FCNModel()
    detector = FCNDetector(vgg.model)
    detector.train(dataset)


if __name__ == '__main__':
    main()
