import cv2
import numpy as np
import matplotlib.pyplot as plt
from recognition import Items, Video
from tools import Display, Geometry, Similarity, Hog


def main(video_input: str):
    svm = cv2.ml.SVM_load("mats\\item_recognize.data")
    camera = Video.Camera(6, 30, 53, 30)
    background_subtractor = cv2.createBackgroundSubtractorKNN()
    video = Video.Video(cv2.VideoCapture(video_input), video_name=video_input)
    shape = (video.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
             video.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    rect_y = shape[0] // 12 * 3
    rect_x = shape[1] // 12 * 2
    boundary = Geometry.Rect(rect_x, rect_y, shape[1] // 12 * 8, shape[0] // 12 * 6)
    item_count = 0
    th_hog = None

    for i in range(video.total_frames_num):
        success, frame = video.video_capture.read()
        if not success:
            break
        frame = Video.Frame(frame, i + 1, video)
        copy = frame.cv_frame.copy()

        frame_blur = cv2.GaussianBlur(frame.cv_frame.copy(), (13, 13), 0)  # 高斯模糊
        fgmask = background_subtractor.apply(frame_blur)  # 由KNN产生
        th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]  # 二值化
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)  # 扩张
        # cv2.imshow("dilated", dilated)
        contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        Items.Item.overlap_match(video.items)

        for c in contours:
            if shape[0] * shape[1] / 2 > cv2.contourArea(c) > 3000:
                (x, y, w, h) = cv2.boundingRect(c)
                if Geometry.Rect.has_inside(boundary, Geometry.Rect(x, y, w, h).get_mid_point()):
                    if w / h >= 2:
                        continue
                    # cv2.rectangle(frame.cv_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
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
                                if similarity[0] >= 0.5 and 0.3 < each.rect.size() / (w * h) < 3.33 \
                                        and 0.6 < (each.rect.width/each.rect.height) / (w/h) < 1.67:
                                    each.tracker = cv2.TrackerMOSSE_create()
                                    each.tracker.init(frame.cv_frame, (x, y, w, h))
                                    each.remain = True
                                elif similarity < 0.2:
                                    time = video.get_time(i)
                                    item = Items.Item(item_count, x, y, w, h, time, frame.cv_frame)
                                    item.remain = True
                                    item.is_overlapping = True
                                    video.items.append(item)
                                    item_count += 1
                                    cv2.rectangle(copy, (x, y), (x + w, y + h), (255, 255, 0), 2)
                                    break
                                break
                    else:
                        time = video.get_time(i)
                        item = Items.Item(item_count, x, y, w, h, time, frame.cv_frame)
                        item.remain = True
                        video.items.append(item)
                        item_count += 1
                        cv2.rectangle(copy, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # get updated location of objects in subsequent frames
        for each in video.items:
            success, box = each.update_tracker(frame_blur, camera)
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
                if len(each.trace) % 25 == 0:
                    shot = each.take_quick_shot(frame.cv_frame)

                    # recognition test
                    if shot is not None:
                        hogs = []
                        descriptor = cv2.HOGDescriptor()
                        shot = cv2.resize(shot, (200, 200))
                        gray = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
                        hog = descriptor.compute(gray)
                        hogs.append(hog)
                        hogs = np.array(hogs)
                        _, p = svm.predict(hogs)
                        print(p)

                    each.sort_quick_shots()
                    if len(each.quick_shots) >= 3:
                        pics = []
                        for j in range(3):
                            pics.append(each.quick_shots[j])
                        each.quick_shots = pics

            # item will be killed
            elif len(each.trace) >= 20:
                time = video.get_time(i)
                each.end_time = time
                video.died_items.append(each)
                each.take_quick_shot(frame.cv_frame)
                # make hog
                if th_hog is not None:
                    th_hog.join()
                th_hog = Hog.HogThread(each.quick_shots, [])
                th_hog.start()
                each.display_quick_shots()
        video.items = new_items

        for each in video.items:
            cv2.rectangle(copy, each.rect.get_coord(), each.rect.get_coord_opposite(), (0, 0, 255), 2)
            _, average_speed = each.get_speed(video, i + 1)
            string = "ID: " + str(each.identification) \
                     + " Distance: %.2fm" % each.get_distance(camera, video.picture_rect.height) \
                     + " horizontal: %.2fm" % each.get_horizontal_offset(camera) \
                     + " average speed: %.2fm/s" % average_speed
            cv2.putText(copy, string, each.rect.get_coord(), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

        cv2.rectangle(copy, boundary.get_coord(), boundary.get_coord_opposite(), (255, 255, 0), 2)

        # show frame
        video.add_frame_to_video(copy)
        cv2.imshow('frame', copy)

        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break

    cv2.destroyAllWindows()
    video.save_video_info()
    video.save_dead_items()


if __name__ == '__main__':
    main('video-02.mp4')
