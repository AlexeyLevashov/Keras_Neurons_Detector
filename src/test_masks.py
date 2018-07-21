import cv2
from modules.dataset import Dataset


def show_images(images):
    current_index = 0
    cv2.namedWindow('image', cv2.WINDOW_GUI_EXPANDED)
    while 1:
        cv2.imshow('image', images[current_index])
        key = int(cv2.waitKey())
        if ord('1') <= key <= ord('4'):
            current_index = key - ord('1')
        if key == 32 or key == 13:
            break
        if key == 27:
            exit()


def main():
    TEST_SOURCE_IMAGES = 0
    TEST_BATCH_IMAGES = 1

    dataset = Dataset()

    if TEST_SOURCE_IMAGES:
        for rects_image in dataset.images_data:
            images = [rects_image.image, rects_image.mask[:, :, 0], rects_image.mask[:, :, 1],
                      rects_image.mask[:, :, 2]]
            show_images(images)

    if TEST_BATCH_IMAGES:
        while 1:
            images_batch, masks_batch = dataset.get_batch()
            image = images_batch[0]
            mask = masks_batch[0]
            images = [image, mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]]
            show_images(images)


if __name__ == '__main__':
    main()
