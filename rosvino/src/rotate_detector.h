#ifndef ROTATE_DETECTOR_H
#define ROTATE_DETECTOR_H
#include <chrono>
#include <cmath>
#include <inference_engine.hpp>
#include <iostream>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using namespace InferenceEngine;

class Rotate_Detector
{
  public:
    typedef struct {
        float    prob;
        int      classId;
        cv::Rect rect;
        float    angle;
    } Object;
    Rotate_Detector();
    ~Rotate_Detector();

std::vector<float> anchors = {
	10,13, 16,30, 33,23,
	30,61, 62,45, 59,119,
	116,90, 156,198, 373,326

//        12,16, 19,36, 40,28,
//        36,75, 76,55, 72,146,
//        142,110, 192,243, 459,401

};

    //初始化
    bool init(string xml_path, double cof_threshold, double nms_area_threshold);
    //释放资源
    bool uninit();
    //处理图像获取结果
    bool process_frame(Mat& inframe, vector<Object>& detected_objects);

  private:

int get_anchor_index(int scale_w, int scale_h);
float get_stride(int scale_w, int scale_h);
float sigmoid_function(float a);

    double      sigmoid(double x);
    vector<int> get_anchors(int net_grid);
    bool        parse_yolov5(const Blob::Ptr& blob, int net_grid, float cof_threshold, vector<Rect>& o_rect, vector<float>& o_rect_cof, vector<int>& classIds, vector<float>& rotate_angle);
    Rect        detet2origin(const Rect& dete_rect, float rate_to, int top, int left);
    //存储初始化获得的可执行网络
    ExecutableNetwork _network;
    OutputsDataMap    _outputinfo;
    string            _input_name;
    //参数区
    string _xml_path;           // OpenVINO模型xml文件路径
    double _cof_threshold;      //置信度阈值,计算方法是框置信度乘以物品种类置信度
    double _nms_area_threshold; // nms最小重叠面积阈值

    int col;
    int row;

};
#endif
