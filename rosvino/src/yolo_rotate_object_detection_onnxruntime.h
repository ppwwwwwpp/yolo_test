#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;
using namespace Ort;


class YOLO {
public:

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	float objThreshold;  //Object Confidence threshold
};

typedef struct BoxInfo
{
	RotatedRect box;
	float score;
	int label;
} BoxInfo;


    YOLO();
    ~YOLO();
	//YOLO(Net_config config);
        void yoloConfig(Net_config &config);
	void detect(Mat& frame);
private:
	const float anchors[3][6] = { {10.0,  13.0, 16.0,  30.0,  33.0,  23.0},
								 {30.0,  61.0, 62.0,  45.0,  59.0,  119.0},
								 {116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };
	const float stride[3] = { 8.0, 16.0, 32.0 };
	/*const int inpWidth = 1024;
	const int inpHeight = 1024;*/
	int inpWidth;
	int inpHeight;
	int nout;
	int num_proposal;
	vector<string> class_names;
	int num_class;

	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	const bool keep_ratio = true;
	vector<float> input_image_;
	void normalize_(Mat img);
	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);
	void nms_angle(vector<BoxInfo>& input_boxes);

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "yolov5");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs


};
