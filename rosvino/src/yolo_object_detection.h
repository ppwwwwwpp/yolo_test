#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <inference_engine.hpp>


using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace InferenceEngine;

class YOLOObjectDetection {
public:

        struct Detection
        {
            int class_id;
            float confidence;
            Rect box;
        };


	void detect(std::string xml, std::string bin, cv::Mat color_mat, int camera_index, vector<Detection> &net_outputs);
private:
	void inferAndOutput(cv::Mat &frame, InferenceEngine::InferRequest &request, InferenceEngine::InputsDataMap & input_info, 
		InferenceEngine::OutputsDataMap &output_info, float sx, float sy);


        std::string m_outputName = "";
        vector<Detection> outputs;      
	//vector<Rect> boxes;
	//vector<int> classIds;
	//vector<float> confidences;


};
