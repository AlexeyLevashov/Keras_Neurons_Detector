import cv2
from modules.reader import Dataset


def main():
    dataset = Dataset()
    for rects_image in dataset.images:
        images = [rects_image.image, rects_image.mask[:, :, 0], rects_image.mask[:, :, 1], rects_image.mask[:, :, 2]]
        current_index = 0
        cv2.namedWindow('image', cv2.WINDOW_GUI_EXPANDED)
        while 1:
            cv2.imshow('image', images[current_index])
            key = int(cv2.waitKey())
            if ord('1') <= key <= ord('4'):
                current_index = key - ord('1')
            if key == 10:
                break
            if key == 27:
                exit()


if __name__ == '__main__':
    main()
