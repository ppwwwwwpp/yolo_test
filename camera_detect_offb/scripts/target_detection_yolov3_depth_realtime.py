#!/home/wl/anaconda3/bin/python
# coding=UTF-8

from __future__ import print_function, division

import logging
import os
import sys
from argparse import ArgumentParser, SUPPRESS
from math import exp as exp
from time import time

import cv2
from openvino.inference_engine import IENetwork, IECore

import numpy as np

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError



cv_image=[]
updated = 0


logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


def build_argparser():
 parser = ArgumentParser(add_help=False)
 args = parser.add_argument_group('Options')
 args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
 args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
 default="/home/wl/test_ws/darknet/src/yolov7-ros/weights/best.xml", type=str)
 args.add_argument("-i", "--input", help="Required. Path to a image/video file. (Specify 'cam' to work with "
 "camera)", type=str)
 args.add_argument("-l", "--cpu_extension",
 help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
 "the kernels implementations.", type=str, default=None)
 args.add_argument("-d", "--device",
 help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
 " acceptable. The sample will look for a suitable plugin for device specified. "
 "Default value is CPU", default="MYRIAD", type=str)
 args.add_argument("--labels", help="Optional. Labels mapping file", default=None, type=str)
 args.add_argument("-t", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
 default=0.4, type=float)
 args.add_argument("-iout", "--iou_threshold", help="Optional. Intersection over union threshold for overlapping "
 "detections filtering", default=0.4, type=float)
 args.add_argument("-ni", "--number_iter", help="Optional. Number of inference iterations", default=1, type=int)
 args.add_argument("-pc", "--perf_counts", help="Optional. Report performance counters", default=False,
 action="store_true")
 args.add_argument("-r", "--raw_output_message", help="Optional. Output inference results raw values showing",
 default=False, action="store_true")
 return parser


class YoloV3Params:
 # ------------------------------------------- Extracting layer parameters ------------------------------------------
 # Magic numbers are copied from yolo samples
 def __init__(self, param, side):
 self.num = 6 if 'num' not in param else int(param['num'])
 self.coords = 4 if 'coords' not in param else int(param['coords'])
 self.classes = 7 if 'classes' not in param else int(param['classes'])
 self.anchors = [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

 if 'mask' in param:
 mask = [int(idx) for idx in param['mask'].split(',')]
 #print(mask)
 #mask=[[3, 4, 5], [0, 1, 2]]
 self.num = len(mask)

 maskedAnchors = []
 for idx in mask:
 maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
 self.anchors = maskedAnchors

 self.side = side


 def log_params(self):
 params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
 [log.info(" {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]


def entry_index(side, coord, classes, location, entry):
 side_power_2 = side ** 2
 n = location // side_power_2
 loc = location % side_power_2
 return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
 xmin = int((x - w / 2) * w_scale)
 ymin = int((y - h / 2) * h_scale)
 xmax = int(xmin + w * w_scale)
 ymax = int(ymin + h * h_scale)
 return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)


def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
 # ------------------------------------------ Validating output parameters ------------------------------------------
 _, _, out_blob_h, out_blob_w = blob.shape
 assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
 "be equal to width. Current height = {}, current width = {}" \
 "".format(out_blob_h, out_blob_w)

 # ------------------------------------------ Extracting layer parameters -------------------------------------------
 orig_im_h, orig_im_w = original_im_shape
 resized_image_h, resized_image_w = resized_image_shape
 objects = list()
 predictions = blob.flatten()
 side_square = params.side * params.side

 # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
 for i in range(side_square):
 row = i // params.side
 col = i % params.side
 for n in range(params.num):
 obj_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
 scale = predictions[obj_index]
 if scale < threshold:
 continue
 box_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
 x = (col + predictions[box_index + 0 * side_square]) / params.side * resized_image_w
 y = (row + predictions[box_index + 1 * side_square]) / params.side * resized_image_h
 # Value for exp is very big number in some cases so following construction is using here
 try:
 w_exp = exp(predictions[box_index + 2 * side_square])
 h_exp = exp(predictions[box_index + 3 * side_square])
 except OverflowError:
 continue
 w = w_exp * params.anchors[2 * n]
 h = h_exp * params.anchors[2 * n + 1]
 for j in range(params.classes):
 class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
 params.coords + 1 + j)
 confidence = scale * predictions[class_index]
 if confidence < threshold:
 continue
 objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
 h_scale=orig_im_h / resized_image_h, w_scale=orig_im_w / resized_image_w))
 return objects


def intersection_over_union(box_1, box_2):
 width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
 height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
 if width_of_overlap_area < 0 or height_of_overlap_area < 0:
 area_of_overlap = 0
 else:
 area_of_overlap = width_of_overlap_area * height_of_overlap_area
 box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
 box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
 area_of_union = box_1_area + box_2_area - area_of_overlap
 if area_of_union == 0:
 return 0
 return area_of_overlap / area_of_union



def detect(current_image):

 (rows,cols,channels) = current_image.shape
 cv2.imshow("Image window", current_image)
 #cv2.waitKey(3)

 getimg=current_image.copy()
 lower_blue=np.array([16,0,0])
 upper_blue=np.array([44,255,255])

 getimg = np.zeros([rows, cols, channels], np.uint8)
 hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)

 mask = cv2.inRange(hsv, lower_blue, upper_blue)
 getimg=cv2.add(getimg,current_image,mask=mask)

 # # Read and pre-process input images
 n, c, h, w = net.inputs[input_blob].shape

 is_async_mode = False
 wait_key_code = 0


 cur_request_id = 0
 next_request_id = 1
 render_time = 0
 parsing_time = 0

 request_id = cur_request_id
 in_frame = cv2.resize(getimg, (w, h))

 # resize input_frame to network size
 in_frame = in_frame.transpose((2, 0, 1)) # Change data layout from HWC to CHW
 in_frame = in_frame.reshape((n, c, h, w))

 # Start inference
 start_time = time()
 exec_net.start_async(request_id=request_id, inputs={input_blob: in_frame})
 det_time = time() - start_time

 # Collecting object detection results
 objects = list()
 if exec_net.requests[cur_request_id].wait(-1) == 0:
 output = exec_net.requests[cur_request_id].outputs

 start_time = time()
 for layer_name, out_blob in output.items():
 layer_params = YoloV3Params(net.layers[layer_name].params, out_blob.shape[2])
 log.info("Layer {} parameters: ".format(layer_name))
 layer_params.log_params()
 objects += parse_yolo_region(out_blob, in_frame.shape[2:],
 getimg.shape[:-1], layer_params,
 args.prob_threshold)
 parsing_time = time() - start_time

 # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
 objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
 for i in range(len(objects)):
 if objects[i]['confidence'] == 0:
 continue
 for j in range(i + 1, len(objects)):
 if intersection_over_union(objects[i], objects[j]) > args.iou_threshold:
 objects[j]['confidence'] = 0

 # Drawing objects with respect to the --prob_threshold CLI parameter
 objects = [obj for obj in objects if obj['confidence'] >= args.prob_threshold]

 if len(objects) and args.raw_output_message:
 log.info("\nDetected boxes for batch {}:".format(1))
 log.info(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")

 origin_im_size = getimg.shape[:-1]
 #print(objects)
 for obj in objects:
 # Validation bbox of detected object
 if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
 continue
 color = (int(min(obj['class_id'] * 12.5+100, 255)),
 min(obj['class_id'] * 7+80, 255), min(obj['class_id'] * 5+60, 255))
 det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
 str(obj['class_id'])

 if args.raw_output_message:
 log.info(
 "{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(det_label, obj['confidence'], obj['xmin'],
 obj['ymin'], obj['xmax'], obj['ymax'],
 color))

 cv2.rectangle(getimg, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
 print("detect"+det_label)
 cv2.putText(getimg,
 "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
 (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

 # Draw performance stats over frame
 inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
 "Inference time: {:.3f} ms".format(det_time * 1e3)
 render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1e3)
 async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
 "Async mode is off. Processing request {}".format(cur_request_id)
 parsing_message = "YOLO parsing time is {:.3f}".format(parsing_time * 1e3)

 cv2.putText(getimg, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
 cv2.putText(getimg, render_time_message, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
 cv2.putText(getimg, async_mode_message, (10, int(origin_im_size[0] - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
 (10, 10, 200), 1)
 cv2.putText(getimg, parsing_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)

 start_time = time()
 cv2.imshow("DetectionResults", getimg)
 render_time = time() - start_time

 cv2.waitKey(3)
 if is_async_mode:
 cur_request_id, next_request_id = next_request_id, cur_request_id
 getimg = next_frame



def load_model():
 global args,model_xml,model_bin,ie,net,input_blob,exec_net,labels_map
 args = build_argparser().parse_args()

 model_xml = args.model
 model_bin = os.path.splitext(model_xml)[0] + ".bin"

 # ------------- 1. Plugin initialization for specified device and load extensions library if specified -------------
 log.info("Creating Inference Engine...")
 ie = IECore()
 # if args.cpu_extension and 'CPU' in args.device:
 # ie.add_extension(args.cpu_extension, "CPU")

 # -------------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) --------------------
 log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
 net = IENetwork(model=model_xml, weights=model_bin)

 assert len(net.inputs.keys()) == 1, "Sample supports only YOLO V3 based single input topologies"

 # ---------------------------------------------- 4. Preparing inputs -----------------------------------------------
 log.info("Preparing inputs")
 input_blob = next(iter(net.inputs))

 # Defaulf batch_size is 1
 net.batch_size = 1

 # Read and pre-process input images
 n, c, h, w = net.inputs[input_blob].shape

 if args.labels:
 with open(args.labels, 'r') as f:
 labels_map = [x.strip() for x in f]
 else:
 labels_map = None

 # ----------------------------------------- 5. Loading model to the plugin -----------------------------------------
 log.info("Loading model to the plugin")
 exec_net = ie.load_network(network=net, num_requests=2, device_name=args.device)


def doit():
 rate = rospy.Rate(100)
 global cv_image, updated

 while not rospy.is_shutdown():
 if updated == 1:
 detect(cv_image)
 updated = 0
 rate.sleep()


def my_callback(data):
 global cv_image, updated
 bridge = CvBridge()
 #rospy.loginfo(rospy.get_caller_id() + "I heard %s")
 cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
 updated = 1
 # print(str(rr.shape))
 # cv2.imshow("ggggg",cv_image)
 # cv2.waitKey(1)


def my_listener():
 rospy.init_node('image_converter', anonymous=True)
 image_sub = rospy.Subscriber("/camera/color/image_raw",Image,my_callback)
 doit()


if __name__ == '__main__':
 global cv_image
 load_model()
 my_listener()

