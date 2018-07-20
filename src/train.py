from modules.reader import Dataset
from modules.detector import FCNDetector
import modules.models.vgg as vgg_module


def main():
    dataset = Dataset()
    vgg = vgg_module.FCNModel.build_model()
    detector = FCNDetector(vgg)
    detector.train(dataset)


if __name__ == '__main__':
    main()
