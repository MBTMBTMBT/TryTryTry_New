# coding=utf8
import cv2 as cv
import os
from math import *
from tools import *
from recognition import Items

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


class Frame(object):

    def __init__(self, cv_frame, serial_num, video):
        self.cv_frame = cv_frame
        self.serial_num = serial_num
        self.video = video
        self.milliseconds = serial_num / video.fps * 1000

    def __str__(self):
        rst = "Frame num %d, %d, %s" % (self.serial_num, self.milliseconds, Display.format_time(self.milliseconds))
        return rst


class Camera(object):

    def __init__(self, camera_height: float, visual_angle_vertical: int, visual_angle_horizontal: int, depression_angle: int):
        self.camera_height = camera_height
        self.visual_angle_vertical = radians(visual_angle_vertical)
        self.visual_angle_horizontal = radians(visual_angle_horizontal)
        self.depression_angle = radians(depression_angle)
        self.vision_height = self.count_vision_height()

    def count_vision_height(self):
        h = self.camera_height
        a = self.visual_angle_vertical
        b = self.depression_angle
        return 2 * sin(b / 2) * h / sin(a + b / 2)

    def count_distance(self, x: float):
        h = self.camera_height
        a = self.visual_angle_vertical
        b = self.depression_angle
        m = sin(a + b / 2)
        # print(2 * ((h / m) ** 2), cos(radians(90) - (b / 2)))
        numerator = 2 * ((h / m) ** 2) - 2 * x * h * cos(radians(90) - b / 2) / m
        denominator = 2 * h / m * sqrt(x ** 2 + (h / m) ** 2 - 2 * x * h * cos(radians(90) - b / 2) / m)
        theta = a + b / 2 - acos(numerator / denominator)
        # print(numerator)
        # print(denominator)
        # print(theta)
        return h / tan(theta)

    def count_relative_height(self, pixel: int, pixel_height=1080):
        return pixel / pixel_height * self.vision_height

    def count_horizontal_offset(self, distance, pixel, pixel_width=1920):
        hypotenuse = sqrt(self.camera_height ** 2 + distance ** 2)
        width = 2 * tan(self.visual_angle_horizontal / 2) * hypotenuse
        horizontal_offset = pixel / pixel_width * width - width / 2
        return horizontal_offset


class Video(object):

    def __init__(self, video_capture, video_name='untitled'):
        if not video_capture.isOpened():
            raise OSError("文件打开失败！")
        self.video_capture = video_capture
        self.fps = int(video_capture.get(cv.CAP_PROP_FPS))
        self.picture_rect \
            = Geometry.Rect(0, 0, video_capture.get(cv.CAP_PROP_FRAME_WIDTH),
                            video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.total_frames_num = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))
        self.items = []
        self.died_items = []
        size = (int(self.picture_rect.width), int(self.picture_rect.height))
        index = video_name.find('.')
        if index == -1:
            result_name = video_name + "-result.avi"
        else:
            result_name = video_name[0:index] + "-result.avi"
        self.video_name = video_name[0:index]
        path = "." + os.sep + self.video_name
        try:
            os.mkdir(path)
        except WindowsError:
            pass
        self.video_write \
            = cv.VideoWriter('.\\%s\\' % self.video_name + result_name,
                             cv.VideoWriter_fourcc('M', 'P', '4', '2'), self.fps, size)
        # 'I', '4', '2', '0'

    def add_item(self, item: Items.Item):
        self.items.append(item)

    def add_frame_to_video(self, cv_frame):
        self.video_write.write(cv_frame)

    def __str__(self):
        return "frame num: %d, fps: %d, size: %d * %d" \
               % (self.total_frames_num, self.fps, self.picture_rect.width, self.picture_rect.height)

    def save_video_info(self):
        time_length = self.total_frames_num / self.fps * 1000
        time_length = Display.format_time(time_length)
        file = open('%s\\video_info.txt' % self.video_name, 'w')
        file.write("* 视频名；%s\n" % self.video_name)
        file.write("* 总时长；%s, FPS：%s\n" % (time_length, self.fps))
        file.write("* 总通过数：%d\n" % len(self.died_items))
        file.close()

    def save_dead_item(self, item):
        # file = open('%s\\%d.txt' % (self.video_name, item.identification), 'w')
        count = 0
        for each_img in item.quick_shots:
            string = "%s\\%d_%d.png" % (self.video_name, item.identification, count)
            cv.imwrite(string, each_img)
            count += 1
        file = open('%s\\id%d.txt' % (self.video_name, item.identification), 'w')
        file.write("* 出现时间；%dms %s\n" % (item.start_time, Display.format_time(item.start_time)))
        file.write("* 离开时间：%dms %s\n" % (item.end_time, Display.format_time(item.end_time)))
        start_point = item.real_trace[0]
        end_point = item.real_trace[-2]
        file.write("* 位移距离：%.2fm\n"
                   % sqrt((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2))
        file.write("* 平均速度：%.2fm/s\n" % item.average_speed)
        # file.write("%f, %f; %f, %f\n" % (end_point[0], end_point[1], start_point[0], start_point[1]))

    def save_dead_items(self):
        for each in self.died_items:
            self.save_dead_item(each)

    def get_time(self, frame_count: int):
        return frame_count / self.fps * 1000


if __name__ == '__main__':
    camera = Camera(6, 30, 53, 30)
    print(camera.count_relative_height(540))
    print(camera.count_distance(2))
