from modules.geometry import Rect


class Rect2RectTransform:
    def __init__(self, rect_from, rect_to):
        self.cx = rect_from.x
        self.cy = rect_from.y
        self.scale_x = rect_from.w / rect_to.w
        self.scale_y = rect_from.h / rect_to.h
        self.target_rect = rect_to

    def get_pos(self, x, y):
        x -= self.cx
        y -= self.cy
        x *= self.scale_x
        y *= self.scale_y
        return x, y

    def transform_rect(self, rect):
        output_rect = Rect()
        output_rect.x, output_rect.y = self.get_pos(rect.x, rect.y)
        output_rect.w *= self.scale_x
        output_rect.h *= self.scale_y
        return output_rect
