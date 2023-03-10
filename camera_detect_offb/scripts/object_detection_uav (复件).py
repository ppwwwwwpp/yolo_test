#!/home/wl/anaconda3/bin/python
# coding=UTF-8
"""
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0 /usr/bin/env python

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function, division

import logging
import os
import sys
from argparse import ArgumentParser, SUPPRESS
from math import exp as exp
from time import time
import time as tm   
import numpy as np
import rospy
from sensor_msgs.msg import Image   
from cv_bridge import CvBridge,CvBridgeError

#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import threading
from openvino.inference_engine import IENetwork, IECore 

from geometry_msgs.msg import PoseStamped

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


def callback(data):
    ## 将图像通过np读取为数组
    img = np.ndarray(shape=(data.height, data.width, 3),
                dtype=np.uint8, buffer=data.data)
    ## cv2默认图像为BGR，但是其他为RGB，进行颜色空间转换
    # frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame = img
    # print("Callback Process!")
    ## 计算回调一次的时间
    time1 = time()
    calculate(frame)
    # print("Callback Time: ",time()-time1)
    cv2.waitKey(1)


def showImage():
    ## 从话题中读取图像信息，并进行回调
    rospy.init_node('Image_subscriber', anonymous = True)
    rospy.Subscriber('/usb_cam/image_raw', Image, callback, queue_size = 1, buff_size=524288000)
    # print("Subscribe Image Process!")
    rospy.spin()


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input", help="Required. Path to an image/video file. (Specify 'cam' to work with "
                                            "camera)", required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
                           "the kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Labels mapping file", default=None, type=str)
    args.add_argument("-t", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    args.add_argument("-iout", "--iou_threshold", help="Optional. Intersection over union threshold for overlapping "
                                                       "detections filtering", default=0.4, type=float)
    args.add_argument("-ni", "--number_iter", help="Optional. Number of inference iterations", default=1, type=int)
    args.add_argument("-pc", "--perf_counts", help="Optional. Report performance counters", default=False,
                      action="store_true")
    args.add_argument("-r", "--raw_output_message", help="Optional. Output inference results raw values showing",
                      default=False, action="store_true")
    return parser


class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        if 'mask' in param:
            mask = [int(idx) for idx in param['mask'].split(',')]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

        self.side = side
        self.isYoloV3 = 'mask' in param  # Weak way to determine but the only one.


    def log_params(self):
        params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
        [log.info("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]


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
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + predictions[box_index + 0 * side_square]) / params.side
            y = (row + predictions[box_index + 1 * side_square]) / params.side
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
            h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h, w_scale=orig_im_w))
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


def calculate(data):
    global is_async_mode
    global cur_request_id
    global w
    global h
    global n
    global c
    global exec_net
    global input_blob
    global net
    global args
    global labels_map
    global render_time
    global wait_key_code
    global parsing_time
    global next_request_id

    start_time_while = time()
    target_corner = PoseStamped()

    yolo_target_pub = rospy.Publisher('yolo_target_corner', PoseStamped, queue_size=1)

    rate = rospy.Rate(30)
    ## 采用异步的模式
    if is_async_mode:
        next_frame = data    
    else:
        frame = data

    if is_async_mode: 
        request_id = next_request_id
        ## 对图像重新定义长度和宽度，降低图像的像素
        in_frame = cv2.resize(next_frame, (w, h))
    else:
        request_id = cur_request_id
        in_frame = cv2.resize(frame, (w, h))
    
    # resize input_frame to network size
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW

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
            out_blob = out_blob.reshape(net.layers[net.layers[layer_name].parents[0]].shape)
            layer_params = YoloParams(net.layers[layer_name].params, out_blob.shape[2])
            log.info("Layer {} parameters: ".format(layer_name))
            layer_params.log_params()
            objects += parse_yolo_region(out_blob, in_frame.shape[2:],
            #changed
                                        # frame.shape[:-1], layer_params,
                                        next_frame.shape[:-1], layer_params,
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

    # changed
    # origin_im_size = frame.shape[:-1]
    origin_im_size = next_frame.shape[:-1]

    for obj in objects:
        # Validation bbox of detected object
        if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
            continue
        color = (255,255,0)
        det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
            str(obj['class_id'])

        if args.raw_output_message:
            log.info(
                "{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(det_label, obj['confidence'], obj['xmin'],
                                                                        obj['ymin'], obj['xmax'], obj['ymax'],
                                                                        color))

        # changed
        # cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
        # cv2.putText(frame,
        #             "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
        #             (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
        cv2.rectangle(next_frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
        cv2.putText(next_frame,
                    "#" + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
                    (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)


    objects = sorted(objects, key=lambda obj : obj['class_id'], reverse=False)  ##sort lambda 函数，按class_id值从小到大排列
    try: 
        ## QuaternionStamped.x, y, z, w = xmin, ymin, xmax, ymax
        obj = objects[0]
        target_corner.pose.orientation.x = obj['xmin']
        target_corner.pose.orientation.y = obj['ymin']
        target_corner.pose.orientation.z = obj['xmax']
        target_corner.pose.orientation.w = obj['ymax']
        target_corner.header.stamp = rospy.Time.now()
    except:
        print("Failed")

    yolo_target_pub.publish(target_corner)
    rate.sleep()


    # Draw performance stats over frame
    inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
        "Inference time: {:.3f} ms".format(det_time * 1e3)
    render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1e3)
    async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
        "Async mode is off. Processing request {}".format(cur_request_id)
    parsing_message = "YOLO parsing time is {:.3f}".format(parsing_time * 1e3)

    # changed
    # cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
    # cv2.putText(frame, render_time_message, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
    cv2.putText(next_frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
    cv2.putText(next_frame, render_time_message, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)

    # changed
    # cv2.putText(frame, parsing_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
    cv2.putText(next_frame, parsing_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)    

    if is_async_mode:
        cur_request_id, next_request_id = next_request_id, cur_request_id
        frame = next_frame

    start_time = time()
    cv2.imshow("DetectionResults", frame)
    render_time = time() - start_time

    key = cv2.waitKey(wait_key_code)

    # Tab key
    if key == 9:
        exec_net.requests[cur_request_id].wait()
        is_async_mode = not is_async_mode
        log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))
        
    end_time_while = time() - start_time_while
    # print("time_while =", end_time_while)


def main():
    global args

    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.cpu_extension = False
    args.model = '/home/wl/test_ws/darknet/src/yolov7-ros/weights/best.xml'
    args.bin = '/home/wl/test_ws/darknet/src/yolov7-ros/weights/best.bin'
    args.device = 'MYRIAD'
    args.labels = '/home/wl/test_ws/darknet/src/yolov7-ros/weights/best.mapping'
    args.input = 'cam'
    args.prob_threshold = 0.3
    args.iou_threshold = 0.3
    args.raw_output_message = True
    args.no_show = False

    # ------------- 1. Plugin initialization for specified device and load extensions library if specified -------------
    log.info("Creating Inference Engine...")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")

    # -------------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) --------------------
    log.info("Loading network")
    global net
    # net = IENetwork(model=model_xml, weights=model_bin)
    #net = ie.read_network(args.model,args.bin)
    net = ie.read_network(model=args.model, weights=args.bin)

    # ---------------------------------- 3. Load CPU extension for support specific layer ------------------------------
    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                        format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                        "or --cpu_extension command line argument")
            sys.exit(1)

    assert len(net.inputs.keys()) == 1, "Sample supports only YOLO V3 based single input topologies"

    # ---------------------------------------------- 4. Preparing inputs -----------------------------------------------
    log.info("Preparing inputs")
    global input_blob
    input_blob = next(iter(net.inputs))

    #  Defaulf batch_size is 1
    net.batch_size = 1

    # Read and pre-process input images
    global w
    global h
    global n
    global c
    global labels_map

    n, c, h, w = net.inputs[input_blob].shape

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    input_stream = 0 if args.input == "cam" else args.input

    global is_async_mode
    is_async_mode = True

    global wait_key_code
    wait_key_code = 1

    # Number of frames in picture is 1 and this will be read in cycle. Sync mode is default value for this case
    # if number_input_frames != 1:
    #     ret, frame = cap.read()
    # else:
    #     is_async_mode = False
    #     wait_key_code = 0

    # ----------------------------------------- 5. Loading model to the plugin -----------------------------------------
    log.info("Loading model to the plugin")
    global exec_net
    #exec_net = ie.load_network(network=net, num_requests=2, device_name=args.device)
    exec_net = ie.load_network(network=net, device_name='CPU', num_requests=2)
    global cur_request_id
    global render_time
    cur_request_id = 0
    global parsing_time
    global next_request_id
    next_request_id = 1
    render_time = 0
    parsing_time = 0

    # ----------------------------------------------- 6. Doing inference -----------------------------------------------
    log.info("Starting inference...")
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    print("To switch between sync/async modes, press TAB key in the output window")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    ## 调用showImage，里面有回调callback
    showImage()
