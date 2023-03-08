#!/home/wl/anaconda3/bin/python
# coding=UTF-8


from __future__ import division, print_function

import math
import os
import pickle
import socket
import sys
from argparse import SUPPRESS, ArgumentParser
from datetime import datetime
from math import exp as exp
from time import time

#sys.path.append('/opt/intel/openvino_2019.2.242/python/python3.5')
#sys.path.append("/opt/intel/openvino_2019.2.242/python/python3")

from openvino.inference_engine import IENetwork, IEPlugin

#sys.path.remove('/opt/intel/openvino_2019.2.242/python/python3.5')
#sys.path.remove("/opt/intel/openvino_2019.2.242/python/python3")

import cv2
import numpy as np

#from robot.helpers import recv_obj
from helpers import recv_obj

# sys.path.append("/opt/intel/openvino/inference_engine/lib/intel64")

# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')



def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input", help="Required. Path to a image/video file. (Specify 'cam' to work with "
                                            "camera, 'server' for realsense server)", required=True, type=str)
    args.add_argument("-pp", "--plugin_dir", help="Optional. Path to a plugin folder", type=str, default=None)
    args.add_argument("-t", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    args.add_argument("-iout", "--iou_threshold", help="Optional. Intersection over union threshold for overlapping "
                                                       "detections filtering", default=0.4, type=float)
    args.add_argument("-ni", "--number_iter", help="Optional. Number of inference iterations", default=1, type=int)
    args.add_argument("-pc", "--perf_counts", help="Optional. Report performance counters", default=False,
                      action="store_true")

    return parser


class YoloV3Params:
    # ------------------------------------------- Extracting layer parameters --------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else len(param['mask'].split(',')) if 'mask' in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]
        self.side = side
        if self.side == 13:
            self.anchor_offset = 2 * 6
        elif self.side == 26:
            self.anchor_offset = 2 * 3
        elif self.side == 52:
            self.anchor_offset = 2 * 0
        else:
            assert False, "Invalid output size. Only 13, 26 and 52 sizes are supported for output spatial dimensions"


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
    # ------------------------------------------ Validating output parameters --------------------------------------
    _, _, out_blob_h, out_blob_w = blob.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters ---------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    predictions = blob.flatten()
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output ---------------------------------------
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
            w = w_exp * params.anchors[params.anchor_offset + 2 * n]
            h = h_exp * params.anchors[params.anchor_offset + 2 * n + 1]
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

def intersection_over_box2(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_2_area 
    if area_of_overlap == 0:
        return 0
    return area_of_overlap / area_of_union


def load_model(device, labels, model_xml, model_bin, plugin_dir,
               cpu_extension='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'):
    # ------------- 1. Plugin initialization for specified device and load extensions library if specified ----------
    plugin = IEPlugin(device=device, plugin_dirs=plugin_dir)
    if 'CPU' in device:
        plugin.add_cpu_extension(cpu_extension)

    # -------------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) ----------------
    net = IENetwork(model=model_xml, weights=model_bin)

    # ---------------------------------- 3. Load CPU extension for support specific layer --------------------------
    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            print("Following layers are not supported by the plugin for specified device {}:\n {}".
                  format(plugin.device, ', '.join(not_supported_layers)))
            print("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                  "or --cpu_extension command line argument")
            sys.exit(1)

    assert len(net.inputs.keys()) == 1, "Sample supports only YOLO V3 based single input topologies"
    assert len(net.outputs) == 3, "Sample supports only YOLO V3 based triple output topologies"

    # ---------------------------------------------- 4. Preparing inputs -------------------------------------------

    #  Defaulf batch_size is 1
    net.batch_size = 1

    # Read and pre-process input images

    return net, plugin.load(network=net, num_requests=2)


def corners2center(xmin, xmax, ymin, ymax):
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    h = ymax - ymin
    w = xmax - xmin
    return x, y, w, h


def recv_object(client):
    packets = []
    while True:
        packet = client.recv(1024)
        if not packet:
            break
        packets.append(packet)
    object = pickle.loads(b"".join(packets))
    return object


def get_frame(input_stream, HOST=None, PORT=None):
    if input_stream == "cam":
        input_stream = 0
        cap = cv2.VideoCapture(input_stream)
        ret, frame = cap.read()
    elif input_stream == "server":
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.connect((HOST, PORT))
            frame = recv_obj(client)
        #RGBD
    else:
        input_stream
        cap = cv2.VideoCapture(input_stream)
        ret, frame = cap.read()
    return frame


def diag(box_1):
    dist1 = ((box_1["xmax"] - box_1["xmin"]) ** 2 + (box_1["ymax"] - box_1["ymin"]) ** 2) ** 0.5
    return dist1


def expected_len(box_1, depth):
    dist = ((depth * math.tan(math.radians((box_1["xmax"] - box_1["xmin"]) * 0.06796))) ** 2 + (
                depth * math.tan(math.radians((box_1["ymax"] - box_1["ymin"]) * 0.0806))) ** 2) ** 0.5
    return dist


def prettyprint(detections):
    string = "[\n"
    for i in detections:
        string = string + "    {\n"
        for person_k, person_v in i.items():

            if person_k == "box" or person_k == "equip":
                if person_k == "box":
                    person_v = [person_v]
                string = string + "        [\n"
                for j in person_v:
                    string = string + "            {\n"
                    # print(j)
                    for item_k, item_v in j.items():
                        string = string + "                " + item_k + ": " + str(item_v) + "\n"
                    string = string + "            },\n"
                string = string + "        ]\n"
            else:
                string = string + "        " + person_k + ": " + str(person_v) + "\n"
        string = string + "    },\n"
    string = string + "]"
    print(string)


def detect(rgb, d, net, exec_net, labels_map, prob_thresh, iou_thresh, depth_given=True):
    # if depth_given:
    #     rgb, d = frame[:, :, :3].astype(np.uint8), frame[:, :, 3]
    #     rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    #     frame = np.array(rgb)
    depth = np.array(d)
    v_deg_per_pix = 42.5 / 720
    h_deg_per_pix = 69.4 / 1280

    frame = np.array(rgb)
    # d = np.array(d)

    input_blob = next(iter(net.inputs))
    n, c, h, w = net.inputs[input_blob].shape
    request_id = 0
    in_frame = cv2.resize(frame, (w, h))

    # resize input_frame to network size
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((n, c, h, w))

    exec_net.start_async(request_id=request_id, inputs={input_blob: in_frame})

    # Collecting object detection results
    objects = list()
    if exec_net.requests[request_id].wait(-1) == 0:
        output = exec_net.requests[request_id].outputs

        for layer_name, out_blob in output.items():
            layer_params = YoloV3Params(net.layers[layer_name].params, out_blob.shape[2])
            objects += parse_yolo_region(out_blob, in_frame.shape[2:],
                                         frame.shape[:-1], layer_params,
                                         prob_thresh)

        # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            if intersection_over_union(objects[i], objects[j]) > iou_thresh:
                objects[j]['confidence'] = 0

        # Drawing objects with respect to the --prob_threshold CLI parameter
    objects = [obj for obj in objects if obj['confidence'] >= prob_thresh]

    original_image = frame
    origin_im_size = frame.shape[:-1]
    people = []
    others = []
    for obj in objects:
        # Validation bbox of detected object
        if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
            continue
        color = (int(min(obj['class_id'] * 12.5, 255)), min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
        det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
            str(obj['class_id'])

        detection = {}
        detection["label"] = det_label
        detection["box"] = obj

        labels_to_save = set(["Face", "Rifle", "Handgun", "Knife"])

        if detection["label"] == "Person":
            detection["equip"] = []
            detection["danger_score"] = 0
            detection["depth"] = depth[int((obj["ymax"] + obj["ymin"]) / 2)][int((obj["xmax"] + obj["xmin"]) / 2)] / 1000
            detection["h_angle"] = round((((obj["xmax"] + obj["xmin"]) / 2) - 640) * (87 / 1280), 2)
            people.append(detection)
        else:
            if detection["label"] in labels_to_save:
                imgymin = obj["ymin"]-10 if obj["ymin"]-10 > 0 else obj["ymin"]
                imgymax = obj["ymax"]+10 if obj["ymax"]+10 < frame.shape[1] else obj["ymax"]
                imgxmax = obj["xmax"]+10 if obj["xmax"]+10 < frame.shape[0] else obj["xmax"]
                imgxmin = obj["xmin"]-10 if obj["xmin"]-10 < 0 else obj["xmin"]
                detection["image"] = original_image[imgymin:imgymax,imgxmin:imgxmax][...,::-1]
                print(depth[imgymin:imgymax,imgxmin:imgxmax])
            others.append(detection)
            cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
            # cv2.putText(frame, det_label, (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            cv2.putText(frame, det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %' , (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

    object_len = {"Handgun": 0.2, "Hat": 0.2, "Jacket": 0.8, "K nife": 0.1, "Rifle": 0.7, "Sunglasses": 0.05,
                  "Police": 1.7, "Face": 0.2}

    danger_weights = {"Handgun": 5, "Hat": 1, "Jacket": 1, "Knife": 3, "Person": 0, "Rifle": 5, "Sunglasses": 1,
                      "Police": -6, "Face": 0}

    for i in others:
        min_diff = 10000000000
        count = 0
        likely = -1
        for j in range(len(people)):
            if intersection_over_box2(i["box"], people[j]["box"]) > 0:
                est_diff = ((expected_len(i["box"], people[j]["depth"]) - object_len[i["label"]]) ** 2) * intersection_over_box2(i["box"], people[j]["box"])
                print(i["label"], est_diff, expected_len(i["box"], people[j]["depth"]) - object_len[i["label"]])
                if people[j]["depth"] != 0:
                    if min_diff > est_diff and est_diff < 15:
                        min_diff = est_diff 
                        likely = j
                if count < 1 and est_diff < 16:
                    likely = j
                count = count + 1
        if likely != -1:
            people[likely]["equip"].append(i)
            people[likely]["danger_score"] = people[likely]["danger_score"] + i["box"]["confidence"] * danger_weights[
                i["label"]]

    for i in people:
        if i["danger_score"] > 1:
            color = (0,0,255)
        elif i["danger_score"] > 0.2:
            color = (0,165,255)
        else:
            color = (0,255,0)
        obj = i["box"]
        cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
        # cv2.putText(frame, i["label"], (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
        cv2.putText(frame, i["label"] + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %' + ' ' + str(i["depth"]) + " m", (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

    return frame, people


def main():
    args = build_argparser().parse_args()

    # PARAMS
    HOST = "127.0.0.1"
    PORT = 4445

    device = 'CPU'  # GPU
    labels = './custom.names'  # set to None if no labels
    cpu_extension = '/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'
    model_xml = '/home/wl/test_ws/darknet/src/yolov7-ros/weights/best.xml'
    model_bin = '/home/wl/test_ws/darknet/src/yolov7-ros/weights/best.bin'

    # ------------------------------------------- Loading model to the plugin -------------------------------------
    net, exec_net = load_model(device, labels, model_xml, model_bin, plugin_dir=None, cpu_extension=cpu_extension)
    print("Model loaded")

    # mapping labels
    with open(labels, 'r') as f:
        labels_map = [x.strip() for x in f]

    frame = get_frame(args.input, HOST, PORT)

    detections, time = detect(frame, net, exec_net, labels_map, args.prob_threshold, args.iou_threshold,
                              depth_given=True)

    prettyprint(detections)


if __name__ == '__main__':
    sys.exit(main() or 0)
