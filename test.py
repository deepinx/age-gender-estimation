import face_model
import argparse
import cv2
import sys
import numpy as np
import datetime


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
  size = cv2.getTextSize(label, font, font_scale, thickness)[0]
  x, y = point
  cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
  cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--image', default='sample-images/test1.jpg', help='')
# parser.add_argument('--image', default='sample-images/Tom_Hanks_54745.png', help='')
parser.add_argument('--model', default='./model/m1/model,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=1, type=int, help='mtcnn or essh option, 0 means mtcnn, 1 means essh')
args = parser.parse_args()

model = face_model.FaceModel(args)
#img = cv2.imread('Tom_Hanks_54745.png')
img = cv2.imread(args.image)
img_db, bbox, points = model.get_input(img, args)
#f1 = model.get_feature(img)
#print(f1[0:10])
for _ in range(1):
  gender, age = model.get_ga(img_db)
# time_now = datetime.datetime.now()
# count = 200
# for _ in range(count):
#   gender, age = model.get_ga(img_db)
# time_now2 = datetime.datetime.now()
# diff = time_now2 - time_now
# print('time cost', diff.total_seconds()/count)
# if gender==0:
#   print('gender is female')
# else:
#   print('gender is male')
# print('age is %d' % age)

# b = bbox
for b in bbox:
  cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
for p in points:
  for i in range(5):
    cv2.circle(img, (p[i][0], p[i][1]), 1, (0, 0, 255), 2)
for i in range(len(age)):
  label = "{}, {}".format(int(age[i]), "F" if gender[i] == 0 else "M")
  draw_label(img, (int(bbox[i,0]), int(bbox[i,1])), label)
cv2.imshow("detection result", img)
cv2.waitKey(0)

