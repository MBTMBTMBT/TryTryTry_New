# coding=utf8
import cv2 as cv
import math
import os
from tools import *
from recognition import clpr_entry


class Item(object):

    # active_items = dataStructure.LinkedList()
    # inactive_items = dataStructure.LinkedList()

    # 定义每个被追踪物体的参数：ID、长度、宽度、平均速度、被跟丢次数等
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
        self.plates = {}
        self.predicted_plate = ""
        self.plate_count = 0
        self.plate_strs = []

    # def get_location(self) -> Geometry.Point:
    #     return self.location

    # 得到指定物体长、宽、大小
    def get_height(self) -> int:
        return self.rect.height

    def get_width(self) -> int:
        return self.rect.width

    def get_size(self) -> int:
        return self.rect.size()

    # 设定指定物体长度宽度
    def set_height(self, height: int):
        self.rect.height = height

    def set_width(self, width: int):
        self.rect.width = width

    # 根据每一帧获得的图像，更新追踪器参数
    def update_tracker(self, frame, camera):
        success, box = self.tracker.update(frame)

        # 假如追踪器没有跟上移动物体，则lost_times + 1
        # 假如追踪器已经超过两次追踪到的移动物体，则可以通过物体的移动方向预测其下一帧的位置
        # 如果追踪器跟丢次数小于10，则可以通过预测移动物体位置的方式，让追踪器寻找移动物体
        # 假如跟丢次数过多，则舍弃当前跟踪器
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

        # 得到每个追踪器的当前帧下的位置、长度、宽度、追踪路径长度
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

    # 通过移动物体的目前移动距离、当前时间及移动物体出现时间计算该物体当前平均速度
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

    # 判断所追踪的物体是否在移动
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

    # 将拍摄得到的照片按照片大小排序
    def sort_quick_shots(self):
        self.quick_shots.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)

    # 在画面中展示拍摄到的照片
    def display_quick_shots(self):
        count = 0
        for each in self.quick_shots:
            string = "%d %d" % (self.identification, count)
            cv.imshow(string, each)
            count += 1

    # 得到物体与摄像机的竖直高度距离
    def get_distance(self, camera, pixel_height=1080):
        relative_height = camera.count_relative_height(pixel_height - self.rect.get_coord_opposite()[1])
        return camera.count_distance(relative_height)

    # 得到物体与摄像机间的水平偏移量
    def get_horizontal_offset(self, camera, pixel_width=1920, pixel_height=1080):
        pixel = self.rect.get_mid_point().get_coord()[0]
        return camera.count_horizontal_offset(self.get_distance(camera, pixel_height), pixel, pixel_width)

    # 记录物体消除的时间
    def suicide(self, time_in_ms: int):
        self.end_time = time_in_ms

    def record_plate_recognition(self, image, target_video_path=None) -> (bool, str):
        success, rst, roi = Item.predict_plate(image)
        if success:
            # self.plates.append(rst)
            if rst in self.plates.keys():
                self.plates[rst] += 1
            else:
                self.plates[rst] = 1
            if target_video_path is not None:
                try:
                    os.mkdir(target_video_path + "\\id%d-plate" % self.identification)
                except WindowsError:
                    pass
                cv.imwrite(target_video_path + "\\id%d-plate\\plate-%d.png"
                           % (self.identification, self.plate_count), roi)
                self.plate_count += 1
                self.plate_strs.append(rst)
        max_num = 0
        max_key = rst
        # print(self.plates)
        for key in self.plates.keys():
            if self.plates[key] > max_num:
                max_num = self.plates[key]
                max_key = key
        else:
            self.predicted_plate = max_key
        print(self.predicted_plate)
        return success, rst

    # 识别车牌 - 静态方法
    @staticmethod
    def predict_plate(image) -> (bool, str):
        try:
            rst, roi = clpr_entry.clpr_main(image)
            # if rst == '':
            #     return False, rst, roi
            return True, rst, roi
        except:
            return False, '', None

    # 避免由于两个物体重叠时调整追踪器矩形框大小而跟丢当前被追踪物体
    # 因此设置“are_overlapping”参数，若其为“True”状态，则该物体追踪器矩形框与其他物体追踪器矩形框有重叠；反之亦然
    @staticmethod
    def are_overlapping(item1, item2) -> bool:
        return Geometry.Rect.are_overlapping(item1.rect, item2.rect)

    # 重置每个物体重叠状态
    @staticmethod
    def set_all_not_overlapping(items: []):
        for each in items:
            each.is_overlapping = False

    # 匹配两个碰撞物体，若两个物体相互重叠，则都设成重叠状态
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
