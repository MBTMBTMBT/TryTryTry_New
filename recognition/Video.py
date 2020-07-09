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

    # 定义每一帧参数；所属视频及出现时间
    def __init__(self, cv_frame, serial_num, video):
        self.cv_frame = cv_frame
        self.serial_num = serial_num
        self.video = video
        self.milliseconds = serial_num / video.fps * 1000

    def __str__(self):
        rst = "Frame num %d, %d, %s" % (self.serial_num, self.milliseconds, Display.format_time(self.milliseconds))
        return rst


class Camera(object):

    # 定义相机相关参数：相机所处高度；水平、竖直视角范围及俯角
    def __init__(self, camera_height: float, visual_angle_vertical: int, visual_angle_horizontal: int, depression_angle: int):
        self.camera_height = camera_height
        # 将角度值转化成弧度制表示
        self.visual_angle_vertical = radians(visual_angle_vertical)
        self.visual_angle_horizontal = radians(visual_angle_horizontal)
        self.depression_angle = radians(depression_angle)
        self.vision_height = self.count_vision_height()

    # 通过计算相机所处高度及其视角范围计算拍摄画面中物体的实际高度
    def count_vision_height(self):
        h = self.camera_height
        a = self.visual_angle_vertical
        b = self.depression_angle
        return 2 * sin(b / 2) * h / sin(a + b / 2)

    # 通过相机参数及画面中物体与标准点距离的数值计算，得到实际物体与相机的水平距离
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

    # 计算物体在画面中的像素高度
    def count_relative_height(self, pixel: int, pixel_height=1080):
        return pixel / pixel_height * self.vision_height

    # 计算物体在画面中的斜边偏移量
    def count_horizontal_offset(self, distance, pixel, pixel_width=1920):
        hypotenuse = sqrt(self.camera_height ** 2 + distance ** 2)
        width = 2 * tan(self.visual_angle_horizontal / 2) * hypotenuse
        horizontal_offset = pixel / pixel_width * width - width / 2
        return horizontal_offset


class Video(object):

    # 生成最终指定视频文件
    def __init__(self, video_capture, video_name='untitled'):
        # 假如无法打开原视频文件，则报错
        if not video_capture.isOpened():
            raise OSError("文件打开失败！")
        # 设定生成视频参数：画面大小、帧率等
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
        # 其文件名为：“原视频文件名”+"-result.avi"
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
        # 指定生成视频格式
        self.video_write \
            = cv.VideoWriter('.\\%s\\' % self.video_name + result_name,
                             cv.VideoWriter_fourcc('M', 'P', '4', '2'), self.fps, size)
        # 'I', '4', '2', '0'

    # 将当前状态下活跃的物体添加到视频列表中
    def add_item(self, item: Items.Item):
        self.items.append(item)

    # 向生成的视频中写入准备好的帧
    def add_frame_to_video(self, cv_frame):
        self.video_write.write(cv_frame)

    def __str__(self):
        return "frame num: %d, fps: %d, size: %d * %d" \
               % (self.total_frames_num, self.fps, self.picture_rect.width, self.picture_rect.height)

    # 在指定文件中存储视频参数：储存路径、名称、时长、总通过车辆数及被追踪物体参数
    def save_video_info(self):
        time_length = self.total_frames_num / self.fps * 1000
        time_length = Display.format_time(time_length)
        file = open('%s\\video_info.txt' % self.video_name, 'w')
        file.write("* 视频名；%s\n" % self.video_name)
        file.write("* 总时长；%s, FPS：%s\n" % (time_length, self.fps))
        file.write("* 总通过数：%d\n" % len(self.died_items))
        for each in self.died_items:
            file.write("id%d\n" % each.identification)
        file.close()

    # 在指定文件中存储被追踪的出现并离开画面的移动物体(dead_item)参数：
    # 总数量、出现时间、离开时间、位移距离及平均速度
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
        file.write("* 平均速度：%.2fm/s <-> %.2fkm/h\n" % (item.average_speed, item.average_speed * 3.6))
        file.close()
        try:
            file = open('%s\\id%d-plate\\plates.txt' % (self.video_name, item.identification), 'w')
            count = 0
            for plate_str in item.plate_strs:
                file.write("%d: %s\n" % (count, plate_str))
                count += 1
        except FileNotFoundError:
            pass
        # file.write("%f, %f; %f, %f\n" % (end_point[0], end_point[1], start_point[0], start_point[1]))

    # 对于每个出现在被追踪到的画面中并离开画面的移动物体，记录他们的各个参数
    def save_dead_items(self):
        for each in self.died_items:
            self.save_dead_item(each)

    # 得到某一帧的相对时间
    def get_time(self, frame_count: int):
        return frame_count / self.fps * 1000


if __name__ == '__main__':
    camera = Camera(6, 30, 53, 30)
    print(camera.count_relative_height(540))
    print(camera.count_distance(2))
