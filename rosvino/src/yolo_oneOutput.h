#ifndef YOLO_ONEOUTPUT_H
#define YOLO_ONEOUTPUT_H
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <iostream>
#include <thread>
#include <fstream>
#include <chrono>
#include <opencv2/dnn/dnn.hpp>
#include <cmath>
using namespace std;
using namespace cv;
using namespace InferenceEngine;

class yoloOneOutput
{
public:
    typedef struct {
        int id;
        float prob;
        cv::Rect rect;
    } Object;
    yoloOneOutput();
    ~yoloOneOutput();
    bool init(string xml_path, string bin_path, double cof_threshold, double nms_area_threshold);
    bool process_frame(Mat& inframe, vector<Object> &detected_objects);

private:
    double sigmoid(double x);
    vector<int> get_anchors(int net_grid);
    bool parse_yolov5(const Blob::Ptr &blob, int net_grid, float cof_threshold, vector<Rect>& o_rect, vector<float>& o_rect_cof, vector<int>& classIds);
    ExecutableNetwork _network;
    OutputsDataMap _outputinfo;
    string _input_name;
    string _xml_path;
    string _bin_path;
    double _cof_threshold = 0.1;
    double _nms_area_threshold = 0.5;
};
#endif
