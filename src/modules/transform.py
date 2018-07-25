from modules.geometry import Rect


class Rect2RectTransform:
    def __init__(self, rect_from, rect_to):
        self.cx = rect_from.x
        self.cy = rect_from.y
        self.scale_x = rect_from.w / rect_to.w
        self.scale_y = rect_from.h / rect_to.h
        self.w = rect_to.w
        self.h = rect_to.h
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

    def get_rect_inner_area(self, rect):
        transformed_rect = self.transform_rect(rect)
        transformed_rect.x = int(transformed_rect.x)
        transformed_rect.y = int(transformed_rect.y)
        transformed_rect.w = int(transformed_rect.w)
        transformed_rect.h = int(transformed_rect.h)

        inner_area = Rect(0, 0, transformed_rect.w, transformed_rect.h)
        if transformed_rect.x < 0:
            inner_area.x = -transformed_rect.x
            transformed_rect.x = 0
        if transformed_rect.y < 0:
            inner_area.y = -transformed_rect.y
            transformed_rect.y = 0
        if transformed_rect.x + transformed_rect.w > self.w:
            inner_area.w -= (transformed_rect.x + transformed_rect.w) - self.w
        if transformed_rect.y + transformed_rect.h > self.h:
            inner_area.h -= (transformed_rect.y + transformed_rect.h) - self.h

        is_inside = True
        if inner_area.x < -transformed_rect.w / 2 or inner_area.y < -transformed_rect.h / 2:
            is_inside = False
        if inner_area.w < transformed_rect.w / 2 or inner_area.h < transformed_rect.h / 2:
            is_inside = False

        return transformed_rect, inner_area, is_inside
