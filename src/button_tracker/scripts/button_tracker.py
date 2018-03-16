#!/usr/bin/env python
import os
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import CompressedImage
from button_recognition.srv import *

text_coding = {1: '1', 2: '2', 3: '<', 4: '>', 5: '3', 6: '!',
               7: '4', 8: 'blur', 9: '5', 10: 'hazy', 11: '6', 12: '7',
               13: '8', 14: 'special', 15: '0', 16: 'alarm', 17: '9', 18: 'G',
               19: 'B', 20: 'up', 21: 'down', 22: 'call', 23: 'L', 24: 'star',
               25: 'P', 26: '-', 27: 'M', 28: 'stop', 29: 'U', 30: 'D',
               31: 'R', 32: 'A', 33: 'C', 34: 'S', 35: 'E', 36: 'F',
               37: 'O', 38: 'K', 39: 'H', 40: 'N', 41: 'T', 42: 'V',
               43: 'I', 44: 'Z', 45: 'J', 46: 'X', 47: '<null>'}

VIDEO_PATH = '/home/DataBase/elevator_panel_database/samples/sample-2.MOV'


class ButtonTracker:
  def __init__(self):
    self.detected_box = None
    self.tracker = None

  def init_tracker(self, image, box_list):
    self.tracker = None
    self.tracker = cv2.MultiTracker_create()
    for box_item in box_list:
      self.tracker.add(cv2.TrackerKCF_create(), image, box_item)

  @staticmethod
  def call_for_service(image):
    rospy.wait_for_service('recognition_service')
    compressed_image = CompressedImage()
    compressed_image.header.stamp = rospy.Time.now()
    compressed_image.data = np.array(cv2.imencode('.jpg', image)[1]).tostring()
    try:
      recognize = rospy.ServiceProxy('recognition_service', recog_server)
      response = recognize(compressed_image)
      box_list = []
      score_list = []
      class_list = []
      char_list = []
      for box in response.box.data:
        rect = tuple(
          [box.x_min, box.y_min, box.x_max - box.x_min, box.y_max - box.y_min])
        box_list.append(rect)
        score_list.append(box.score)
        class_list.append(box.categ)
        str_disp = ''
        char_0 = text_coding[box.char0 + 1]
        char_1 = text_coding[box.char1 + 1]
        char_2 = text_coding[box.char2 + 1]
        if box.char0 != 46:
          str_disp += char_0
        if box.char1 != 46:
          str_disp += char_1
        if box.char2 != 46:
          str_disp += char_2
        char_list.append(str_disp)
      return box_list, score_list, class_list, char_list
    except rospy.ServiceException, e:
      print "Service call failed: {}".format(e)


def read_video(video_name):
  if not os.path.exists(video_name):
    raise IOError('Invalid video path or device number!')
  video = cv2.VideoCapture(video_name)
  if not video.isOpened():
    rospy.logwarn('Cannot open the video or device!')
    sys.exit()
  rospy.loginfo('Initialize the tracker ...')
  button_tracker = ButtonTracker()
  (state, frame) = video.read()
  img_height = frame.shape[0] / 2
  img_width = frame.shape[1] / 2
  frame = cv2.resize(frame, dsize=(img_width, img_height))
  (boxes, scores, classes, chars) = button_tracker.call_for_service(frame)
  button_tracker.init_tracker(frame, boxes)
  # TODO: replace the counter with an measurement about the tracker
  counter = 0
  while state:
    counter += 1
    (state, frame) = video.read()
    frame = cv2.resize(frame, dsize=(img_width, img_height))
    if not state:
      sys.exit()
    ok, boxes = button_tracker.tracker.update(frame)
    if counter % 10 == 0:
      (boxes, scores, classes, chars) = button_tracker.call_for_service(frame)
      button_tracker.init_tracker(frame, boxes)
    for box, char in zip(boxes, chars):
      p1 = (int(box[0]), int(box[1]))
      p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
      cv2.rectangle(frame, p1, p2, (0, 255, 0), thickness=3)
      cv2.putText(frame, char, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0))
    cv2.imshow('button_tracker', frame)
    k = cv2.waitKey(1)
    if k == 27:
      break  # esc pressed


def read_image(image_list):
  images = []
  for image_name in image_list:
    if not os.path.exists(image_name):
      raise IOError('Image path {} not exist!'.format(image_name))
    frame = cv2.imread(image_name, cv2.IMREAD_COLOR)
    button_tracker = ButtonTracker()
    (boxes, scores, classes, chars) = button_tracker.call_for_service(frame)
    for box, char in zip(boxes, chars):
      p1 = (int(box[0]), int(box[1]))
      p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
      cv2.rectangle(frame, p1, p2, (0, 255, 0), thickness=3)
      text_position = (box[0]+20, box[1]+50)
      cv2.putText(frame, char, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0))
    images.append(frame)
  for idx, item in enumerate(images):
    win_name = 'image_{}'.format(idx)
    cv2.imshow(win_name, item)
  cv2.waitKey(0)


if __name__ == '__main__':
  rospy.init_node('button_tracker', anonymous=True)
  img_only = rospy.get_param('button_tracker/img_only', True)
  if img_only:
    image_path = rospy.get_param('button_tracker/image_path', '../test_samples')
    image_number = rospy.get_param('button_tracker/image_number', 3)
    img_list = [os.path.join(image_path, 'image{}.jpg'.format(i)) for i in range(0, image_number)]
    for img_item in img_list:
      if not os.path.exists(img_item):
        raise IOError('Image path not exist: {}'.format(img_item))
    read_image(img_list)
  else:
    video_path = rospy.get_param('button_tracker/video_path', '../test_samples/sample-4.MOV')
    if not os.path.exists(video_path):
      raise IOError('Video path not exist!')
    read_video(video_path)
  rospy.loginfo('Process Finished!')
