import cv2
from tools import Similarity


class Tracker(object):

    def __init__(self, frame, bounding_box):
        self.tracker = cv2.TrackerMOSSE_create()
        self.frame = frame
        self.x = bounding_box[0]
        self.y = bounding_box[1]
        self.w = bounding_box[2]
        self.h = bounding_box[3]
        self.image = frame[self.x:self.x + self.w, self.y:self.y + self.h]
        self.tracker.init(self.frame, (self.x, self.y, self.w, self.h))
        self.pics = []
        self.get_zoomed_pics()

    def zoom(self, gain: float):
        width = self.w * gain
        height = self.w * gain
        coord_x = self.x - (width - self.w) / 2
        coord_y = self.y - (height - self.h) / 2
        width = int(width)
        height = int(height)
        coord_x = int(coord_x)
        coord_y = int(coord_y)
        try:
            image = self.frame[coord_x:coord_x + width, coord_y, coord_y + height]
        except:
            image = None
        return image

    def get_zoomed_pics(self):
        pics = []
        for i in range(-5, 6):
            i = 1 + i / 10
            pics.append(cv2.resize(self.zoom(i), (self.w, self.h)))
        self.pics = pics

    def update(self, frame):
        self.frame = frame
        success, box = self.tracker.update(frame)
        if not success:
            return success, box
        self.image = self.frame[box[0]:box[0] + box[2], box[1]:box[1] + box[3]]
        similarity = []
        for each in self.pics:
            if each is None:
                similarity.append(0)
            else:
                similarity.append(Similarity.classify_hist_with_split(each, self.image, (self.w, self.h)))
        max_similarity = 0
        max_index = 0
        for i in range(len(similarity)):
            if similarity[i] > max_similarity:
                max_similarity = similarity[i]
                max_index = i
        gain = (max_index - 5) / 10 + 1
        gain = 1 / gain
        w = self.w
        h = self.h
        self.image = self.zoom(gain)
        self.w *= gain
        self.h *= gain
        self.x = self.x - (w - self.w) / 2
        self.y = self.y - (h - self.h) / 2
        self.get_zoomed_pics()
        return success, (self.x, self.y, self.w, self.h)
