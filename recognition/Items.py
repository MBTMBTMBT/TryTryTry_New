# coding=utf8
import cv2 as cv
import math
from tools import *
from recognition import clpr_entry


class Item(object):

    # active_items = dataStructure.LinkedList()
    # inactive_items = dataStructure.LinkedList()

    def __init__(self, identification: int, coord_x: int, coord_y: int,
                 width: int, height: int, start_time: int, cv_frame):
        self.identification = identification
        # self.location = Geometry.Point(coord_x, coord_y)
        self.rect = Geometry.Rect(coord_x, coord_y, width, height)
        self.break_count = 0
        self.tracker = cv.TrackerMOSSE_create()
        self.tracker.init(cv_frame, (coord_x, coord_y, width, height))
        self.trace = []
        self.real_trace = []
        self.move_length = []
        self.speed = []
        self.average_speed = 0
        self.is_overlapping = False
        self.remain = True
        self.lost_times = 0
        self.quick_shots = []
        self.take_quick_shot(cv_frame)
        self.start_time = start_time
        self.end_time = 0

    # def get_location(self) -> Geometry.Point:
    #     return self.location

    def get_height(self) -> int:
        return self.rect.height

    def get_width(self) -> int:
        return self.rect.width

    def get_size(self) -> int:
        return self.rect.size()

    # def set_location(self, coord_x: int, coord_y: int) -> Geometry.Point:
    # self.location = Geometry.Point(coord_x, coord_y)
    # return self.location

    def set_height(self, height: int):
        self.rect.height = height

    def set_width(self, width: int):
        self.rect.width = width

    def update_tracker(self, frame, camera):
        success, box = self.tracker.update(frame)
        if not success:
            # print("I fucked up!")
            self.lost_times += 1
            if len(self.trace) >= 2:
                box = [0, 0, 0, 0]
                # print(self.trace[len(self.trace) - 1][0] * 2 - self.trace[len(self.trace) - 2][0])
                if self.lost_times <= 10:
                    box[0] = self.trace[len(self.trace) - 1][0] * 2 - self.trace[len(self.trace) - 2][0]
                    box[1] = self.trace[len(self.trace) - 1][1] * 2 - self.trace[len(self.trace) - 2][1]
                    box[2] = self.trace[len(self.trace) - 1][2]
                    box[3] = self.trace[len(self.trace) - 1][3]
            else:
                self.lost_times = 0
        self.rect.location = Geometry.Point(int(box[0]), int(box[1]))
        self.rect.width = int(box[2])
        self.rect.height = int(box[3])
        self.trace.append(box)
        rect = Geometry.Rect(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        distance = self.get_distance(camera, frame.shape[0])
        horizontal_offset \
            = self.get_horizontal_offset(camera, frame.shape[1], rect.get_mid_point().get_coord()[0])
        self.real_trace.append((distance, horizontal_offset))
        if len(self.real_trace) > 1:
            now = self.real_trace[len(self.real_trace) - 1]
            last = self.real_trace[len(self.real_trace) - 2]
            length = math.sqrt((now[0] - last[0]) ** 2 + (now[1] - last[1]) ** 2)
            # print(now[0], last[0])
            self.move_length.append(length)
        return success, box

    def get_speed(self, video, frame_count):
        if len(self.move_length) > 2:
            distance = self.move_length[-2]
            speed = distance * video.fps
            self.speed.append(speed)
            now = self.real_trace[-1]
            last = self.real_trace[0]
            total_distance = math.sqrt((now[0] - last[0]) ** 2 + (now[1] - last[1]) ** 2)
            '''
            if self.identification == 25:
                print(self.identification)
                print(now, last)
                print(total_distance, frame_count / video.fps)
                print("================")
                '''
            self.average_speed = total_distance / (frame_count / video.fps - self.start_time / 1000)
            # print(total_distance)
            return speed, self.average_speed
        else:
            return 0, 0

    def is_moving(self):
        try:
            x, y, _, _ = self.trace[len(self.trace) - 10]
            x1, y1, _, _ = self.trace[len(self.trace) - 1]
            return x1 != x or y1 != y
        except IndexError:
            return True

    def take_quick_shot(self, cv_frame):
        x = self.rect.get_coord()[0]
        y = self.rect.get_coord()[1]
        w = self.get_width()
        h = self.get_height()
        # print(x, y, self.get_width(), self.get_height())
        if x >= 0 and y >= 0 and w > 0 and h > 0:
            cut = cv_frame[y: y + h, x: x + w]
            self.quick_shots.append(cut)
            return cut

    def sort_quick_shots(self):
        self.quick_shots.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)

    def display_quick_shots(self):
        count = 0
        for each in self.quick_shots:
            string = "%d %d" % (self.identification, count)
            cv.imshow(string, each)
            count += 1

    def get_distance(self, camera, pixel_height=1080):
        relative_height = camera.count_relative_height(pixel_height - self.rect.get_coord_opposite()[1])
        return camera.count_distance(relative_height)

    def get_horizontal_offset(self, camera, pixel_width=1920, pixel_height=1080):
        pixel = self.rect.get_mid_point().get_coord()[0]
        return camera.count_horizontal_offset(self.get_distance(camera, pixel_height), pixel, pixel_width)

    def suicide(self, time_in_ms: int):
        self.end_time = time_in_ms

    @staticmethod
    def predict_plate(image) -> (bool, str):
        try:
            rst = clpr_entry.clpr_main(image)
            if rst == '':
                return False, rst
            return True, rst
        except:
            return False, ''

    @staticmethod
    def are_overlapping(item1, item2) -> bool:
        return Geometry.Rect.are_overlapping(item1.rect, item2.rect)

    @staticmethod
    def set_all_not_overlapping(items: []):
        for each in items:
            each.is_overlapping = False

    @staticmethod
    def overlap_match(items: []):
        for a in items:
            for b in items:
                if a is b:
                    continue
                # elif a.is_overlapping and b.is_overlapping:
                #     continue
                else:
                    if Item.are_overlapping(a, b):
                        a.is_overlapping = True
                        b.is_overlapping = True

    def __str__(self):
        return "ID: %d, Item of rect with " % self.identification + str(self.rect)


class MatchingItem(Item):

    def __init__(self, coord_x: int, coord_y: int, width: int, height: int):
        super().__init__(coord_x, coord_y, width, height)
        self.last = None
        self.next = None

    def match(self, items: []):

        """
        基础版：
        1. 找重叠的
        2. 如果有多个，连最近的
        :param items:
        :return:
        """

        for each_item in items:
            if Item.are_overlapping(self, each_item):
                if Geometry.Point.distance(self.rect.get_mid_point(), each_item.rect.get_mid_point()) \
                        > (self.rect.width + self.rect.height) / 2:
                    continue
                if self.last is None:
                    self.last = each_item
                    each_item.next = self
                    distance = Geometry.Point.distance(self.rect.get_mid_point(), each_item.rect.get_mid_point())
                elif Geometry.Point.distance(self.rect.get_mid_point(), each_item.rect.get_mid_point()) < distance:
                    distance = Geometry.Point.distance(self.rect.get_mid_point(), each_item.rect.get_mid_point())
                    self.last = each_item
                    each_item.next = self

    def __str__(self):
        num = 0
        if self.last is not None:
            num += 1
        if self.next is not None:
            num += 1
        return super().__str__() + " with %d matches." % num
