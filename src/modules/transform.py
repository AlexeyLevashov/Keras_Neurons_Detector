from modules.geometry import Rect


class Rect2RectTransform:
    def __init__(self, rect_from, rect_to):
        self.x = rect_from.x
        self.y = rect_from.y
        self.scale_x = rect_to.w / rect_from.w
        self.scale_y = rect_to.h / rect_from.h
        self.w = rect_to.w
        self.h = rect_to.h
        self.target_rect = rect_to

    def get_pos(self, x, y):
        x -= self.x
        y -= self.y
        x *= self.scale_x
        y *= self.scale_y
        return x, y

    def transform_rect(self, rect):
        output_rect = Rect()
        output_rect.x, output_rect.y = self.get_pos(rect.x, rect.y)
        output_rect.w = rect.w * self.scale_x
        output_rect.h = rect.h * self.scale_y
        return output_rect

    def get_rect_inner_area(self, rect):
        transformed_rect = self.transform_rect(rect)
        center_x = int(transformed_rect.x + transformed_rect.w/2)
        center_y = int(transformed_rect.y + transformed_rect.h/2)

        is_inside = 0 <= center_x < self.w and 0 <= center_y < self.h
        if not is_inside:
            return None, None, is_inside, None

        transformed_rect.x = int(transformed_rect.x)
        transformed_rect.y = int(transformed_rect.y)
        transformed_rect.w = int(transformed_rect.w)
        transformed_rect.h = int(transformed_rect.h)

        x1 = transformed_rect.x
        y1 = transformed_rect.y
        x2 = transformed_rect.x + transformed_rect.w
        y2 = transformed_rect.y + transformed_rect.h

        inner_area = Rect(0, 0, transformed_rect.w, transformed_rect.h)
        if x1 < 0:
            transformed_rect.x = 0
            inner_area.x = -x1
            inner_area.w -= -x1
            transformed_rect.w -= -x1
        if y1 < 0:
            transformed_rect.y = 0
            inner_area.y = -y1
            inner_area.h -= -y1
            transformed_rect.h -= -y1
        if x2 > self.w:
            dw = x2 - self.w
            inner_area.w -= dw
            transformed_rect.w -= dw
        if y2 > self.h:
            dh = y2 - self.h
            inner_area.h -= dh
            transformed_rect.h -= dh

        return transformed_rect, inner_area, is_inside, (x2-x1, y2-y1)
