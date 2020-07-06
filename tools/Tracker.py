import cv2
from tools import Similarity


class Tracker(object):
    GAIN = (0.83, 0.87, 0.91, 0.95, 1, 1.05, 1.1, 1.15, 1.2)
    UPDATE_FREQUENCY = 10

    def __init__(self, frame, bounding_box):
        self.tracker = cv2.TrackerMOSSE_create()
        self.frame = frame
        self.x = bounding_box[0]
        self.y = bounding_box[1]
        self.w = bounding_box[2]
        self.h = bounding_box[3]
        self.image = frame[self.y:self.y + self.h, self.x:self.x + self.w]
        self.tracker.init(self.frame, (self.x, self.y, self.w, self.h))
        self.pics = []
        self.get_zoomed_pics()
        self.update_count = 0

    def zoom(self, gain: float):
        width = self.w * gain
        height = self.h * gain
        coord_x = self.x - (width - self.w) / 2
        coord_y = self.y - (height - self.h) / 2
        width = int(width)
        height = int(height)
        coord_x = int(coord_x)
        coord_y = int(coord_y)
        '''
        try:
            self.die_frame = cv2.rectangle(self.die_frame, (coord_x, coord_y), (coord_x + width, coord_y + height), (0, 255, 0), 2)
        except:
            pass
            '''
        # frame = cv2.rectangle(self.frame.copy(), (coord_x, coord_y), (coord_x + width, coord_y + height), (0, 255, 0), 2)
        # cv2.imshow("pic", frame)
        try:
            image = self.frame[coord_y:coord_y + height, coord_x:coord_x + width]
        except:
            image = None
        return image

    def get_zoomed_pics(self):
        pics = []
        for i in Tracker.GAIN:
            pic = self.zoom(i)
            if pic is not None:
                try:
                    pics.append(cv2.resize(pic, (self.w, self.h)))
                except:
                    pics.append(None)
            else:
                pics.append(None)
        self.pics = pics
        '''
        count = 0
        for each in pics:
            try:
                cv2.imshow(str(count), each)
            except:
                pass
            count += 1
        cv2.waitKey(0)
        '''

    def update(self, frame):
        self.frame = frame
        # self.die_frame = self.frame.copy()
        success, box = self.tracker.update(frame)
        self.update_count += 1
        if not success:
            return success, box
        self.image = self.frame[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2])]
        if self.update_count % Tracker.UPDATE_FREQUENCY == 0:
            self.x = box[0]
            self.y = box[1]
            self.w = box[2]
            self.h = box[3]
            return success, box
        similarity = []
        for each in self.pics:
            if each is None:
                similarity.append(0)
            else:
                similarity.append(Similarity.classify_hist_with_split(each, self.image, (self.w, self.h))[0])
        max_similarity = 0
        max_index = 0
        for i in range(len(similarity)):
            # print(max_similarity)
            if similarity[i] > max_similarity:
                max_similarity = similarity[i]
                max_index = i
        # gain = Tracker.GAIN[len(Tracker.GAIN) - max_index - 1]
        gain = 1 / Tracker.GAIN[max_index]
        w = self.w
        h = self.h
        # last = self.image.copy()
        self.image = self.zoom(gain)
        '''
        try:
            cv2.imshow("last", last)
            cv2.imshow("next", self.image)
        except:
            pass
            '''
        # cv2.waitKey(0)
        self.w *= gain
        self.h *= gain
        self.x = self.x - (self.w - w) / 2
        self.y = self.y - (self.h - h) / 2
        self.w = int(self.w)
        self.h = int(self.h)
        self.x = int(self.x)
        self.y = int(self.y)
        self.get_zoomed_pics()
        self.tracker = cv2.TrackerMOSSE_create()
        self.tracker.init(frame, (self.x, self.y, self.w, self.h))
        '''
        try:
            cv2.imshow("xxx", self.die_frame)
        except:
            pass
            '''
        return success, (self.x, self.y, self.w, self.h)
