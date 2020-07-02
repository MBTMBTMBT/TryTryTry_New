import cv2
import numpy as np
from tools import SVM


# Train
max_id = 40
direction = "video-02\\"
pics = []
hogs = []
samples = []
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
    resize = cv2.resize(each, (150, 150))
    hogs.append(SVM.get_hog(resize))
print(hogs)

count = 0
for each in pics:
    cv2.imshow("pic", each)
    k = cv2.waitKey(0)
    if k == ord('1'):
        print("car")
        labels.append(1)
        samples.append(hogs[count])
    elif k == ord('2'):
        print("not car")
        labels.append(-1)
        samples.append(hogs[count])
    else:
        print("give up")
        # labels.append(-1)
    print(hogs[count])
    print(samples)
    count += 1

hogs = np.array(hogs)
samples = np.array(samples)
labels = np.array(labels)

print(hogs)
print(samples)
print(labels)

svm = SVM.SVM()
try:
    svm.load('mats\\item_recognize_car_only.data')
except Exception:
    pass
svm.train(samples, labels)

svm.save('mats\\item_recognize_car_only.data')
