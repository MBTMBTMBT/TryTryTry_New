import math


class Point(object):

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def get_coord(self) -> ():
        return self.x, self.y

    def __str__(self):
        return "x: %d, y:%d" % (self.x, self.y)

    @staticmethod
    def distance(a, b) -> float:
        x1, y1 = a.get_coord()
        x2, y2 = b.get_coord()
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class Rect(object):

    def __init__(self, coord_x: int, coord_y: int, width: int, height: int):
        self.location = Point(coord_x, coord_y)
        self.width = width
        self.height = height

    def size(self) -> int:
        return int(self.width * self.height)

    def get_coord(self) -> ():
        return int(self.location.x), int(self.location.y)

    def get_coord_opposite(self) -> ():
        return int(self.location.x + self.width), int(self.location.y + self.height)

    def __str__(self):
        return "coord: %s, w: %d, h: %d, size: %d" \
               % (str(self.location), self.width, self.height, self.size())

    def get_mid_point(self) -> Point:
        return Point(self.location.x + self.width // 2, self.location.y + self.height // 2)

    @ staticmethod
    def has_inside(rect, point):
        ax, ay = rect.get_coord()
        dx, dy = rect.get_coord_opposite()
        x, y = point.get_coord()
        return ax <= x <= dx and ay <= y <= dy

    @ staticmethod
    def are_overlapping(rect1, rect2) -> bool:
        '''
            a <- width -> c
                |
                height
                    |
            b <- width -> d
        '''
        startX1 = rect1.get_coord()[0]
        startY1 = rect1.get_coord()[1]
        endX1 = startX1 + rect1.width
        endY1 = startY1 + rect1.height
        startX2 = rect2.get_coord()[0]
        startY2 = rect2.get_coord()[1]
        endX2 = startX2 + rect2.width
        endY2 = startY2 + rect2.height
        return not (endY2 < startY1 or endY1 < startY2 or startX1 > endX2 or startX2 > endX1)


if __name__ == '__main__':
    rect1 = Rect(160, 270, 600, 720)
    rect2 = Rect(200, 300, 104, 93)
    rect3 = Rect(100, 100, 30, 30)
    print(rect1)
    print(rect2)
    print(rect3)
    print("1 and 2 is overlapping:" + str(is_overlapping(rect1, rect2)))
    print("2 and 3 is overlapping:" + str(is_overlapping(rect2, rect3)))
    print("1 and 3 is overlapping:" + str(is_overlapping(rect1, rect3)))
