#ifndef YOLOV5VINO_H
#define YOLOV5VINO_H
#include <fstream>
#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <inference_engine.hpp>
#define NOT_NCS2
using namespace cv;
using namespace dnn;
using namespace std;
using namespace InferenceEngine;
 
class YOLOVINO
{
public:
    struct Detection
    {
        int class_id;
        float confidence;
        Rect box;
    };
public:
    YOLOVINO();
    ~YOLOVINO();
    void init();
    void loadNet(bool is_cuda);
    Mat formatYolov5(const Mat &source);
    void detect(Mat &image,vector<Detection> &outputs);
    void drawRect(Mat &image,vector<Detection> &outputs);
	void loadClassList();
private:
    float m_scoreThreshold = 0.25; //0.6
    float m_nmsThreshold = 0.5;   //0.6
    float m_confThreshold = 0.92;  //0.8 0.92
	
    const std::string m_classfile = "/home/wl/test_ws/swarm_ws/src/rosvino/models/classes.txt";
    //const std::string m_modelFilename = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/best_s_sim.xml";
    //const std::string m_modelFilename = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/ref_best_sfu.xml";
    const std::string m_modelFilename = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/ref_best_sfujia.xml";//youxian
    //const std::string m_modelFilename = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/ref_best_n.xml";
    //const std::string m_modelFilename = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/rotate_ref_best.xml";
    size_t m_numChannels = 0;
    size_t m_inputH = 0;
    size_t m_inputW = 0;
    size_t m_imageSize = 0;
    std::string m_inputName = "";
    std::string m_outputName = "";
    vector<std::string> m_classNames;
    const vector<Scalar> colors = { Scalar(255, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 0) };
 
    InferRequest m_inferRequest;
    Blob::Ptr m_inputData;



};
#endif
