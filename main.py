import cv2
from recognition import Items, Video
from tools import Display, Geometry, Similarity, Hog


# 摄像机的高度，视角和俯角 - 参考值
CAMERA_HEIGHT = 6
CAMERA_VERTICAL_ANGLE = 20
CAMERA_HORIZONTAL_ANGLE = 38
CAMERA_DEPRESSION_ANGLE = 45

# 识别区的左上坐标和宽、高 - 比例值，不是真实值
BOUNDARY_X = 12 / 0.0001
BOUNDARY_Y = 12 / 2
BOUNDARY_WIDTH = 12 / 12
BOUNDARY_HEIGHT = 12 / 8

# 进行车牌识别的识别线
# 即 - 在此线下方的截图才会进行车牌识别
# 注意这是一个关于屏幕高度的比例值
PLATE_RECOGNITION_LINE = 0.25

# 最小判定帧数 - 即一个物体存在的帧数超过此值才被判定为真正存在
SMALLEST_FRAME_NUMBER_LIMIT = 10

# 用于对不处于“保护态”的物体进行大小更新，以及对相邻区域内的新发现物体做判定
# “保护态” - 是针对处于重叠状态的物体进行的保护设定
#
# 对识别到的物体尺寸做判定使用 - 超出此范围的比例会被认为是误判
RECOGNITION_HEIGHT_WIDTH_RATE_UPPER_LIMIT = 2.5
RECOGNITION_HEIGHT_WIDTH_RATE_LOWER_LIMIT = 0.4
#
# 计算的相似度超过此值的物体会被认为是同一物体
SAME_ITEM_RECOGNITION_SIMILARITY = 0.5
#
# 尺寸更新时的宽高变化接受程度
ITEM_TRACE_HEIGHT_WIDTH_CHANGE_RATE = 0.5
#
# 小于此值的物体会被当成新物体进行跟踪
# 注意 - 与上面的相似度是不同的值
DIFFERENT_ITEM_RECOGNITION_SIMILARITY = 0.4
#
# 尺寸变化的接受程度
# 如 - 0.3代表变化后的面积占变化前的0.3-3.33之间被认为有效
SIZE_UPDATE_RATE_ALLOWANCE = 0.3

# 保留的截图数量
QUICK_SHOT_KEEP_NUM = 6

# 截图频率 - 每多少帧截一次
QUICK_SHOT_TAKEN_FREQUENCY = 10


def main(video_input: str):

    camera = Video.Camera(CAMERA_HEIGHT, CAMERA_VERTICAL_ANGLE, CAMERA_HORIZONTAL_ANGLE, CAMERA_DEPRESSION_ANGLE)
    background_subtractor = cv2.createBackgroundSubtractorKNN()
    video = Video.Video(cv2.VideoCapture(video_input), video_name=video_input)
    shape = (video.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
             video.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    rect_y = shape[0] // BOUNDARY_Y
    rect_x = shape[1] // BOUNDARY_X
    boundary = Geometry.Rect(rect_x, rect_y, shape[1] // BOUNDARY_WIDTH, shape[0] // BOUNDARY_HEIGHT)
    item_count = 0

    for frame_count in range(video.total_frames_num):
        success, frame = video.video_capture.read()
        if not success:
            break
        frame = Video.Frame(frame, frame_count + 1, video)
        copy = frame.cv_frame.copy()
        frame_blur = cv2.GaussianBlur(frame.cv_frame.copy(), (13, 13), 0)  # 高斯模糊
        mask = background_subtractor.apply(frame_blur)  # 由KNN产生
        th = cv2.threshold(mask, 244, 255, cv2.THRESH_BINARY)[1]  # 二值化
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)  # 扩张
        # cv2.imshow("dilated", dilated)

        contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        Items.Item.overlap_match(video.items)

        for c in contours:
            if shape[0] * shape[1] / 4 > cv2.contourArea(c) > 3000:
                (x, y, w, h) = cv2.boundingRect(c)
                if not RECOGNITION_HEIGHT_WIDTH_RATE_UPPER_LIMIT > h / w > RECOGNITION_HEIGHT_WIDTH_RATE_LOWER_LIMIT:
                    continue
                if Geometry.Rect.has_inside(boundary, Geometry.Rect(x, y, w, h).get_mid_point()):
                    for each in video.items:
                        coord = each.rect.get_coord()
                        width = each.rect.width
                        height = each.rect.height
                        each_rect = Geometry.Rect(coord[0], coord[1], width, height)
                        box_rect = Geometry.Rect(x, y, w, h)

                        if Geometry.Rect.are_overlapping(each_rect, box_rect):
                            if each.is_overlapping:
                                break
                            else:
                                each_roi = frame_blur[int(coord[1]):int(coord[1] + height), int(coord[0]):int(coord[0] + width)]
                                box_roi = frame_blur[int(y):int(y + h), int(x):int(x + w)]
                                similarity = Similarity.classify_hist_with_split(each_roi, box_roi)
                                if similarity == -1:
                                    break
                                if similarity[0] >= SAME_ITEM_RECOGNITION_SIMILARITY \
                                        and SIZE_UPDATE_RATE_ALLOWANCE < each.rect.size() / (w * h) < 1 / SIZE_UPDATE_RATE_ALLOWANCE \
                                        and ITEM_TRACE_HEIGHT_WIDTH_CHANGE_RATE < (each.rect.width/each.rect.height) / (w/h) < 1 / ITEM_TRACE_HEIGHT_WIDTH_CHANGE_RATE:
                                    each.tracker = cv2.TrackerMOSSE_create()
                                    each.tracker.init(frame.cv_frame, (x, y, w, h))
                                    each.remain = True
                                elif similarity < DIFFERENT_ITEM_RECOGNITION_SIMILARITY:
                                    time = video.get_time(frame_count)
                                    item = Items.Item(item_count, x, y, w, h, time, frame.cv_frame)
                                    item.remain = True
                                    item.is_overlapping = True
                                    video.items.append(item)
                                    item_count += 1
                                    cv2.rectangle(copy, (x, y), (x + w, y + h), (255, 255, 0), 2)
                                    break
                                break
                    else:
                        time = video.get_time(frame_count)
                        item = Items.Item(item_count, x, y, w, h, time, frame.cv_frame)
                        item.remain = True
                        video.items.append(item)
                        item_count += 1
                        cv2.rectangle(copy, (x, y), (x + w, y + h), (255, 255, 0), 2)

        Items.Item.set_all_not_overlapping(video.items)
        # get updated location of objects in subsequent frames
        for each in video.items:
            success, box = each.update_tracker(frame.cv_frame, camera)
            if Geometry.Rect.has_inside(boundary, Geometry.Rect(box[0], box[1], box[2], box[3]).get_mid_point()):
                if frame.serial_num % 10 != 0:
                    each.remain = True
                elif each.is_moving():
                    each.remain = True
                else:
                    each.remain = False
            else:
                each.remain = False

        new_items = []
        for each in video.items:
            if each.remain:
                new_items.append(each)
                if len(each.trace) % QUICK_SHOT_TAKEN_FREQUENCY == 0:
                    shot = each.take_quick_shot(frame.cv_frame)
                    if each.rect.get_mid_point().get_coord()[1] >= video.picture_rect.height * PLATE_RECOGNITION_LINE:
                        plate_success, plate_str = Items.Item.predict_plate(shot)
                        print(plate_success, plate_str)
                    each.sort_quick_shots()
                    if len(each.quick_shots) >= QUICK_SHOT_KEEP_NUM:
                        pics = []
                        for j in range(QUICK_SHOT_KEEP_NUM):
                            pics.append(each.quick_shots[j])
                        each.quick_shots = pics

            # item will be killed
            elif len(each.trace) >= SMALLEST_FRAME_NUMBER_LIMIT:
                time = video.get_time(frame_count)
                each.suicide(time)
                video.died_items.append(each)
                each.take_quick_shot(frame.cv_frame)
                each.display_quick_shots()
        video.items = new_items

        for each in video.items:
            cv2.rectangle(copy, each.rect.get_coord(), each.rect.get_coord_opposite(), (0, 0, 255), 2)
            _, average_speed = each.get_speed(video, frame_count + 1)
            string = "ID: " + str(each.identification) \
                     + " Distance: %.2fm" % each.get_distance(camera, video.picture_rect.height) \
                     + " horizontal: %.2fm" % each.get_horizontal_offset(camera) \
                     + " average speed: %.2fm/s" % average_speed
            cv2.putText(copy, string, each.rect.get_coord(), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

        cv2.rectangle(copy, boundary.get_coord(), boundary.get_coord_opposite(), (255, 255, 0), 2)

        # show frame
        video.add_frame_to_video(copy)
        cv2.imshow('frame', copy)
        # cv2.imshow('blur', frame_blur)
        cv2.imshow('mask', dilated)

        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break

    cv2.destroyAllWindows()
    video.save_video_info()
    video.save_dead_items()


if __name__ == '__main__':
    # main('video-02.mp4')
    main('MAH00057.mp4')
    # main('video-01.avi')
    # main('video-03.avi')
    # main('video-04.MP4')
