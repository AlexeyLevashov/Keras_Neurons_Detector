import numpy as np
import cv2
import threading


class ImagesViewer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.alive = True
        self.images = []
        self.current_index = 0

        self.lock = threading.Lock()

    def set_images(self, images):
        images = [image.clip(0, 255).astype(np.uint8) for image in images]
        with self.lock:
            self.images = images

    def run(self):
        cv2.namedWindow('ImagesViewer', cv2.WINDOW_GUI_NORMAL)
        while self.alive:
            if len(self.images) > 0:
                if self.current_index >= len(self.images):
                    self.current_index = 0
                with self.lock:
                    image = self.images[self.current_index]
                cv2.imshow('ImagesViewer', image)
            else:
                cv2.imshow('ImagesViewer', np.zeros([200, 200]))

            key = int(cv2.waitKey(100))
            if ord('1') <= key <= ord('9'):
                index = key - ord('1')
                if index < len(self.images):
                    self.current_index = index

