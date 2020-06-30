import cv2
import numpy as np
from numpy.linalg import norm  # 线性代数的范数
import os


class SVM:
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()


def get_hog(image):
    descriptor = cv2.HOGDescriptor()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return descriptor.compute(gray)


'''
max_id = 40
direction = "video-02\\"
pics = []
hogs = []
labels = []

for i in range(max_id + 1):
    for j in range(3):
        try:
            name = direction + "%d_%d.png" % (i, j)
            pic = cv2.imread(name)
            if pic is not None:
                pics.append(pic)
        except Exception:
            continue

for each in pics:
    resize = cv2.resize(each, (200, 200))
    hogs.append(get_hog(resize))

for each in pics:
    cv2.imshow("pic", each)
    k = cv2.waitKey(0)
    if k == ord('1'):
        print("car")
        labels.append(1)
    elif k == ord('2'):
        print("bike")
        labels.append(2)
    elif k == ord('3'):
        print("pedestrian")
        labels.append(3)
    else:
        print(None)
        labels.append(4)

hogs = np.array(hogs)
labels = np.array(labels)

svm = SVM()
try:
    svm.load('mats\\item_recognize.data')
except Exception:
    pass
svm.train(hogs, labels)

svm.save('mats\\item_recognize.data')
'''

max_id = 40
direction = "video-02\\"
pics = []
hogs = []

for i in range(max_id + 1):
    for j in range(3):
        try:
            name = direction + "%d_%d.png" % (i, j)
            pic = cv2.imread(name)
            if pic is not None:
                pics.append(pic)
        except Exception:
            continue

print(1)

for each in pics:
    resize = cv2.resize(each, (200, 200))
    hogs.append(get_hog(resize))

print(2)

hogs = np.array(hogs)
svm = cv2.ml.SVM_load("mats\\item_recognize.data")

print(3)

_, p = svm.predict(hogs)
# print(p)

for i in range(len(hogs)):
    # print(p[i][0])
    r = str(p[i][0])
    if r == '1.0':
        name = "car"
    elif r == '2.0':
        name = "bike"
    elif r == '3.0':
        name = "pedestrian"
    else:
        name = 'n'
    name = str(i) + name
    cv2.imshow(name, pics[i])

cv2.waitKey(0)
