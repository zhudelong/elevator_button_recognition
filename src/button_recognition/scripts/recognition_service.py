#!/usr/bin/env python
import os
import sys
import cv2
import time
import rospy
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from button_recognition.msg import recognition
from button_recognition.msg import recog_result
from button_recognition.srv import *
from utils import label_map_util, visualization_utils as vis_util


# Size, in inches, of the output images.
DISP_IMG_SIZE = (12, 8)
NUM_CLASSES = 1
VERBOSE = False


class ButtonRecognition:
  def __init__(self):
    self.session = None
    self.img_key = None
    self.results = []
    self.ocr_rcnn = {}
    self.init_ocr_rcnn()
    rospy.loginfo('OCR-RCNN initialization finished!')

  def init_ocr_rcnn(self):
    print(rospy.get_param_names())
    graph_path = rospy.get_param(
      'button_recognition/graph_path', '../ocr_rcnn_model/frozen_inference_graph.pb')
    label_path = rospy.get_param(
      'button_recognition/label_path', '../model_config/button_label_map.pbtxt')
    image_path = rospy.get_param(
      'button_recognition/image_path', '../test_samples')
    if not os.path.exists(graph_path):
      print(graph_path)
      raise IOError('Invalid graph path!')
    if not os.path.exists(label_path):
      print(label_path)
      raise IOError('Invalid label path!')
    # Load frozen graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(graph_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    self.session = tf.Session(graph=detection_graph)
    self.img_key = detection_graph.get_tensor_by_name('image_tensor:0')
    self.results.append(detection_graph.get_tensor_by_name('detection_boxes:0'))
    self.results.append(detection_graph.get_tensor_by_name('detection_scores:0'))
    self.results.append(detection_graph.get_tensor_by_name('detection_classes:0'))
    self.results.append(detection_graph.get_tensor_by_name('predicted_chars:0'))
    self.results.append(detection_graph.get_tensor_by_name('num_detections:0'))
    # Load label map
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(
      label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    # return parameters
    self.ocr_rcnn['graph_path'] = graph_path
    self.ocr_rcnn['label_path'] = label_path
    self.ocr_rcnn['image_path'] = image_path
    self.ocr_rcnn['detection_graph'] = detection_graph
    self.ocr_rcnn['category_index'] = category_index
    return True

  def clear_session(self):
    if self.session is not None:
      self.session.close()

  def predict(self, request):
    image = request.image
    if len(image.data) == 0:
      rospy.logerr('None image received!')
      return recog_serverResponse(None)
    start = rospy.get_time()
    if VERBOSE:
      print 'received image of type: "%s"' % image.format
    np_arr = np.fromstring(image.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # image_np = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    img_height = image_np.shape[0]
    img_width = image_np.shape[1]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    (boxes, scores, classes, chars, num) = self.session.run(
      self.results, feed_dict={self.img_key: image_np_expanded})
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)
    chars = np.squeeze(chars)
    num = np.squeeze(num)
    if VERBOSE:
      self.visualize_detection_result(
        image_np, boxes, classes, scores, chars, self.ocr_rcnn['category_index'])
    # collect boxes
    recog_resp = recog_result()
    for box, score, label, char in zip(boxes, scores, classes, chars):
      if score < 0.5:
        num -= 1
        continue
      sample = recognition()
      sample.y_min = int(box[0] * img_height)
      sample.x_min = int(box[1] * img_width)
      sample.y_max = int(box[2] * img_height)
      sample.x_max = int(box[3] * img_width)
      sample.score = score    # float
      sample.categ = label    # int
      sample.char0 = char[0]
      sample.char1 = char[1]
      sample.char2 = char[2]
      recog_resp.data.append(sample)
    end = rospy.get_time()
    rospy.loginfo('Recognition finished: {} objects detected using {} seconds!'.format(
      num, end-start))
    return recog_serverResponse(recog_resp)

  @staticmethod
  def visualize_detection_result(image_np, boxes, classes,
                                 scores, chars, category):
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      np.squeeze(boxes),
      np.squeeze(classes).astype(np.int32),
      np.squeeze(scores),
      category,
      max_boxes_to_draw=100,
      use_normalized_coordinates=True,
      line_thickness=5,
      predict_chars=np.squeeze(chars)
    )
    plt.figure(figsize=DISP_IMG_SIZE)
    plt.imshow(image_np)
    time.sleep(1)


def button_recognition_server():
  recognizer = ButtonRecognition()
  rospy.init_node('button_recognition_server')
  service = rospy.Service('recognition_service',
                          recog_server, recognizer.predict)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    recognizer.clear_session()
    rospy.logdebug('Shutting down ROS button recognition module!')
  cv2.destroyAllWindows()


if __name__ == '__main__':
  button_recognition_server()
