#include <ros/ros.h>
#include <iostream>
#include <thread>
#include <fstream>
#include <inference_engine.hpp>
#include <rosvino/Object.h>
#include <rosvino/Objects.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <geometry_msgs/PoseStamped.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "yolo_object_detection.h"
#include "yolo_rotate_object_detection_onnxruntime.h"
#include "yolov5_6vino.h"
#include "yolo_oneOutput.h"
#include "rotate_detector.h"

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "img_similar.h"


using namespace InferenceEngine;
using namespace cv;

//Timing and CPU usgae
double start;
double stop;
float running_avg_latency = 0;
int count = 0;
int latency_counter = 0;
double start_time = 0; 
float fps = 0;

//Node parameters
std::string device;
float confidence_thresh;
std::string network_loc;

cv::Point2f yolo_center;

InferRequest m_inferRequest;
//param
// double cx_color = 338.960065;
// double cy_color = 244.856900;
// double fx_color = 446.8877;
// double fy_color = 446.032355;  //长直边 16cm  对角线长 18.5cm

double cx_color = 341.530207;
double cy_color = 243.635387;
double fx_color = 451.664427;
double fy_color = 450.081841;

//YOLOVINO yolov7vino;
//yoloOneOutput yolovvino; //buxing
Rotate_Detector rotate_detector;

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


//Class containing subscriber publisher and callback
class Detect{
	public:
	Detect(ros::NodeHandle &nh){
    	//Publisher
    	det_results = nh.advertise<rosvino::Objects>("/detect/det_results",1);

    	//Subscriber
	image_sub = nh.subscribe("/usb_cam/image_raw", 1, &Detect::imageCallback, this);


  /*Mat src = imread("/home/wl/test_ws/orb_rotate/DetectRotationAngle/0006.png");
  //imshow("src",src);
  //截取图片中的某一个区域
  int rows = src.rows;
  int cols = src.cols;
  Rect rect(100,100,100,100);
  Mat dst = src(rect);
  //imshow("dst",dst);
  //waitKey(1);
*/

    //Mat rot_img = imread("/home/wl/realsense_d435/cam/mark_jpg/00287.jpg"); //00287

    //Mat rot_dst = rotate(rot_img, 270); //13
    //cv::imshow("rot_dst", rot_dst);
    //waitKey(1);


    //Mat img_1=imread("/home/wl/test_ws/orb_rotate/0001.png");
    //Mat img_2=imread("/home/wl/test_ws/orb_rotate/0002.png");
    Mat img_1=imread("/home/wl/test_ws/orb_rotate/DetectRotationAngle/9050.png");
    Mat img_2=imread("/home/wl/test_ws/orb_rotate/DetectRotationAngle/5.jpg");

    threshold(img_1,img_1,78, 255,THRESH_BINARY);
    threshold(img_2,img_2,20, 255,THRESH_BINARY);


        Mat img_1gray;
	cvtColor(img_1, img_1gray, COLOR_BGR2GRAY);
	//resize
	resize(img_1gray, img_1gray, Size(100, 75), 0, 0, CV_INTER_AREA);
	//展开成一维
	img_1gray = img_1gray.reshape(0,1);

        Mat img_2gray;
	cvtColor(img_2, img_2gray, COLOR_BGR2GRAY);
	//resize
	resize(img_2gray, img_2gray, Size(100, 75), 0, 0, CV_INTER_AREA);
	//展开成一维
	img_2gray = img_2gray.reshape(0, 1);

	//计算基准图与自己的欧氏距离
	float eucDis1_1 = 0;
	for (auto i = 0; i < img_1gray.cols; ++i)
	{
		eucDis1_1 += sqrt(pow(img_1gray.at<uchar>(0, i) - img_1gray.at<uchar>(0, i), 2));
	}

	std::cout << eucDis1_1 << std::endl;

	float eucDis1_2 = 0;
	for (auto i = 0; i < img_1gray.cols;++i)
	{
		eucDis1_2 += sqrt(pow(img_1gray.at<uchar>(0, i) - img_2gray.at<uchar>(0, i), 2));
	}

	std::cout << eucDis1_2 << std::endl;




    // 将图像转成单通道灰度图像
    //cvtColor(img_1, img_1, COLOR_BGR2GRAY);
    //cvtColor(img_2, img_2, COLOR_BGR2GRAY);

    cv::imshow("img_1", img_1);
    waitKey(1);
    cv::imshow("img_2", img_2);
    waitKey(1);

  if (img_1.channels() == 1) {
    int histSize = 256;
    float range[] = { 0, 256 };
    const float * histRange = { range };

    Mat hist1, hist2;

    calcHist(&img_1, 1, 0, Mat(), hist1, 1, &histSize, &histRange, true, false);
    normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());
    calcHist(&img_2, 1, 0, Mat(), hist2, 1, &histSize, &histRange, true, false);
    normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());

    cv::imshow("hist1", hist1);
    waitKey(1);
    cv::imshow("hist2", hist2);
    waitKey(1);

    double dSimilarity = compareHist(hist1, hist2, CV_COMP_CORREL);
    printf("1: %lf\n", dSimilarity);


 /*    Mat hist1, hist2; // 存储直方图计算结果
    const int channels[1] = { 0 };// 通道索引
    float inRanges[2] = { 0,255 }; //
    const float *rangs[1] = { inRanges }; // 像素灰度值范围
    const int bins[1] = { 256 }; // 直方图的维度，即像素灰度值的最大值
    calcHist(&img_1, 1, channels, Mat(), hist1, 1, bins, rangs);
    calcHist(&img_2, 1, channels, Mat(), hist2, 1, bins, rangs);
    
    // 绘制直方图
    int hist_w = 512;
    int hist_h = 400;
    int width = 2;
    Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
    for (int i = 1; i <= hist.rows; i++) {
        rectangle(histImage, Point(width*(i-1),hist_h-1), Point(width*i-1,hist_h - cvRound(hist.at<float>(i-1) / 15)), Scalar(255,255,255),-1);
    }
    // 归一化直方图
    Mat histImageL1 = Mat::zeros(hist_h, hist_w, CV_8UC3);
    Mat hist_L1;
    normalize(hist, hist_L1, 1, 0,NORM_L2,-1,Mat());
    for (int i = 0; i < hist_L1.rows; i++) {
        rectangle(histImageL1, Point(width*(i - 1), hist_h - 1),
                    Point(width*i - 1, hist_h - cvRound(hist_h*hist_L1.at<float>(i - 1)) - 1),
                    Scalar(255, 255, 255), -1);
    }

    double dSimilarity = compareHist(hist1, hist2, CV_COMP_CORREL);
    printf("1: %lf\n", dSimilarity);
*/



  } else {
    cvtColor(img_1, img_1, COLOR_BGR2HSV);
    cvtColor(img_2, img_2, COLOR_BGR2HSV);

    int h_bins = 50;
    int s_bins = 60;
    int histSize[] = { h_bins, s_bins };
    float h_ranges[] = { 0, 255 };
    float s_ranges[] = { 0, 255 };
    const float * ranges[] = { h_ranges, s_ranges };
    int channels[] = { 0, 1 };

    MatND hist1, hist2;
    calcHist(&img_1, 1, channels, Mat(), hist1, 2, histSize, ranges, true, false);
    normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());
    calcHist(&img_2, 1, channels, Mat(), hist2, 2, histSize, ranges, true, false);
    normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());

    cv::imshow("hist1", hist1);
    waitKey(1);
    cv::imshow("hist2", hist2);
    waitKey(1);

    double dSimilarity = compareHist(hist1, hist2, CV_COMP_CORREL);

    printf("3: %lf\n", dSimilarity);
  }


  //compareFacesByHist(img_2, img_1);

  //int hash_score = hashSimilarity(img_1, img_2);
  //cout << "哈希相似度方法: " << hash_score << "  ... 小于等于6 ：两张图片相同，大于等于9：两张图片不同，7-8：两张有轻微变化" << endl;
  //int aHash_score = aHash(img_1, img_2);
  //cout << "pinjun哈希相似度方法: " << aHash_score << "  ... 小于等于6 ：两张图片相同，大于等于9：两张图片不同，7-8：两张有轻微变化" << endl;

    //Mat img1 = cv::imread("/home/wl/test_ws/orb_rotate/0001.png", CV_LOAD_IMAGE_COLOR);
    //Mat img2 = cv::imread("/home/wl/test_ws/orb_rotate/0002.png", CV_LOAD_IMAGE_COLOR);
   Mat img1=imread("/home/wl/test_ws/orb_rotate/DetectRotationAngle/1.jpg", CV_LOAD_IMAGE_COLOR);
   Mat img2=imread("/home/wl/test_ws/orb_rotate/DetectRotationAngle/6.jpg", CV_LOAD_IMAGE_COLOR);

  threshold(img1,img1,78, 255,THRESH_BINARY);
  threshold(img2,img2,20, 255,THRESH_BINARY);




    //Mat img1 = cv::imread(img1, CV_LOAD_IMAGE_COLOR);
    //Mat img2 = cv::imread(img2, CV_LOAD_IMAGE_COLOR);
    int blur1=(img1.size[0] > img1.size[1] ? img1.size[0] : img1.size[1]) / 20;
    int blur2=(img2.size[0] > img2.size[1] ? img2.size[0] : img2.size[1]) / 20;
    blur(img1, img1, Size(blur1, blur1));
    blur(img2, img2, Size(blur2, blur2));

   // cv::imshow("img1", img1);
    //waitKey(1);
    //cv::imshow("img2", img2);
    //waitKey(1);

    set_fp_size(PART_SIZE * 2);
    Mat fp1 = get_fp(img1);
//    char fps1[4][PART_FP_LEN];
    string fps1[4];
//    fps1[0].resize((u_long)PART_FP_LEN);
    get_fp_strs(fp1, fps1);
    Mat fp2 = get_fp(img2);
//    char fps2[4][PART_FP_LEN];
    string fps2[4];
    get_fp_strs(fp2, fps2);
    cout << fps1[0] << fps1[2] << endl;
    cout << (fps1[0] == fps2[0]) << (fps1[1] == fps2[1]) << (fps1[2] == fps2[2]) << (fps1[3] == fps2[3])<< endl;
    double si = calc_similarity(fp1, fp2);
    cout << "similarity: " << si << endl;





	/*int cell_size = 16;                  //16*16的cell
	int angle_num = 8;                   //角度量化为8
	//图像分割为y_cellnum行，x_cellnum列
	int x_num = img_1.cols / cell_size;
	int y_num = img_1.rows / cell_size;
	int bins = x_num * y_num * angle_num;//数组长度

	float* img1_hog = new float[bins];
	memset(img1_hog, 0, sizeof(float) * bins);
	float* img2_hog = new float[bins];
	memset(img2_hog, 0, sizeof(float) * bins);

	hog_hisgram(img_1, img1_hog, cell_size, angle_num);
	hog_hisgram(img_2, img2_hog, cell_size, angle_num);

	float smlrt1 = Similarity(img1_hog, img2_hog, bins);
        cout << "smlrt1: " << smlrt1 << endl;
*/

/*
		//taking input layer info /usb_cam/image_raw_throttled /usb_cam/image_raw
		InputsDataMap input_info(network_reader.getInputsInfo());
		InputInfo::Ptr& input_data = input_info.begin()->second;
		inputLayerName = input_info.begin()->first;

		input_data->setPrecision(Precision::U8);
	        //input_data->setPrecision(Precision::FP32);
		input_data->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
		input_data->getInputData()->setLayout(Layout::NHWC);

		//taking output layer info
		OutputsDataMap output_info(network_reader.getOutputsInfo());
		DataPtr& output_data = output_info.begin()->second;
		outputLayerName = output_info.begin()->first;
		//num_classes = network_reader.getLayerByName(outputLayerName.c_str())->GetParamAsInt("num_classes");
		output_dimension = output_data->getTensorDesc().getDims();
		results_number = output_dimension[2];
		object_size = output_dimension[3];
		//if ((object_size != 7 || output_dimension.size() != 4)) {
		//	ROS_ERROR("There is a problem with output dimension");
		//}

		//Setting output layer settings
		output_data->setPrecision(Precision::FP32);

		//output_data->setLayout(Layout::NCHW);
		//output_data->setLayout(Layout::NC);

		//Load Network to device

	        model_network = ie.LoadNetwork(network_reader, "CPU");

		//Create Inference Request
		inferReq = model_network.CreateInferRequestPtr();
*/
  	}

	//Image matrix to blob conversion
	// /OpenCV mat to blob
	static InferenceEngine::Blob::Ptr mat_to_blob(const cv::Mat &image) {
	    InferenceEngine::TensorDesc tensor(InferenceEngine::Precision::U8,{1, (size_t)image.channels(), (size_t)image.size().height, (size_t)image.size().width},InferenceEngine::Layout::NHWC);
	    return InferenceEngine::make_shared_blob<uint8_t>(tensor, image.data);
	}

	//Image to blob and associate with inference request
	void frame_to_blob(const cv::Mat& image, InferRequest::Ptr& inferReq, const std::string& descriptor) {
	    inferReq->SetBlob(descriptor, mat_to_blob(image));
	}


	void calculatePoseFromBox(const cv::Rect2f &box)
	{
		object_pose_.pose.position.x = -(box.x+box.width/2-cx_color)/fx_color;
		object_pose_.pose.position.y = (box.y+box.height/2-cy_color)/fy_color;
	}

	void calculatePoseFromPoint(const cv::Point2f &center)
	{
		object_pose_.pose.position.x = -(center.x-cx_color)/fx_color;
		object_pose_.pose.position.y = (center.y-cy_color)/fy_color;
	}

	void calculatePoseFromRotatedBox(const cv::RotatedRect &box)
	{
		object_pose_.pose.position.x = -(box.center.x-cx_color)/fx_color;
		object_pose_.pose.position.y = (box.center.y-cy_color)/fy_color;
		// std::cout << "the yaw angle of the box  = " << box.angle << std::endl;
		object_pose_.pose.orientation.w = cos(-box.angle/180*M_PI/2);
		object_pose_.pose.orientation.z = sin(-box.angle/180*M_PI/2);
		object_pose_.pose.orientation.y = 0;
		object_pose_.pose.orientation.x = 0;
	}



struct match_result {
	std::vector<DMatch> good;
	std::vector<DMatch> all;
};
match_result calculate_matches(const Ptr<ORB>& orb, const Ptr<BFMatcher>& bf, const Mat& a, const Mat& b) {
	std::vector<KeyPoint> keypointsA;
	Mat descriptorsA;
	orb->detectAndCompute(a, noArray(), keypointsA, descriptorsA);

	std::vector<KeyPoint> keypointsB;
	Mat descriptorsB;
	orb->detectAndCompute(b, noArray(), keypointsB, descriptorsB);

	std::vector<std::vector<DMatch> > matches;
	bf->knnMatch(descriptorsA, descriptorsB, matches, 2);

	float lowe_ratio = 0.89f;

	match_result result;
	for (const auto& match: matches) {
		if (match.size() != 2) {
			std::cout << "Invalid knnMatch output: match.size() != 2" << std::endl;
			return match_result();
		}
		if (match[0].distance < match[1].distance * lowe_ratio) result.good.push_back(match[0]);
		result.all.push_back(match[0]);
	}

	return result;
}

float similarity_metrics(const match_result& result) {
	return static_cast<float>(result.good.size()) / result.all.size();
}

float estimate_similarity(cv::Mat &images1,cv::Mat &images2) {
	auto orb = ORB::create();
	auto bf = BFMatcher::create();

	//std::vector<similarity_record> similarities;
	//for (size_t i = 0; i < images1.size(); i++) {
		//for (size_t j = i + 1; j < images2.size(); j++) {
			auto matches = calculate_matches(orb, bf, images1, images2);

			//similarities.push_back(
				//similarity_record {
				//		i,
				//		j,
				float similarities=	similarity_metrics(matches);
				//}
			//);
		//}
	//}

	return similarities;
}



void hog_hisgram(InputArray src, float* histogram, int cellsize, int anglenum)
{
	Mat gray, grd_x, grd_y;                           //灰度，x方向和y方向的梯度
	//计算像素梯度的幅值和方向
	cvtColor(src, gray, COLOR_BGR2GRAY);
	Mat angle, mag;                                   //梯度方向，梯度幅值
	Sobel(gray, grd_x, CV_32F, 1, 0, 3);
	Sobel(gray, grd_y, CV_32F, 0, 1, 3);
	cartToPolar(grd_x, grd_y, mag, angle, true);

	//计算cell的个数
	//图像分割为y_cellnum行，x_cellnum列
	int x_cellnum, y_cellnum;
	x_cellnum = gray.cols / cellsize;
	y_cellnum = gray.rows / cellsize;


	int angle_area = 360 / anglenum;                  //每个量化级数所包含的角度数
	//外循环，遍历cell
	for (int i = 0; i < y_cellnum; i++)
	{
		for (int j = 0; j < x_cellnum; j++)
		{
			//定义感兴趣区域roi,取出每个cell
			Rect roi;
			roi.width = cellsize;
			roi.height = cellsize;
			roi.x = j*cellsize;
			roi.y = i*cellsize;

			Mat RoiAngle, RoiMag;
			RoiAngle = angle(roi);                    //每个cell中的梯度方向
			RoiMag = mag(roi);                        //每个cell中的梯度幅值

			//遍历RoiAngel和RoiMat
			int head = (i * x_cellnum + j) * anglenum;//cell梯度直方图的第一个元素在总直方图中的位置
			for (int m = 0; m < cellsize; m++)
			{
				for (int n = 0; n < cellsize; n++)
				{
					int idx = ((int)RoiAngle.at<float>(m, n)) / angle_area;//该梯度所处的量化级数
					histogram[head + idx] += RoiMag.at<float>(m, n);
				}
			}
		}
	}
	
}




bool compareFacesByHist(Mat img,Mat orgImg)
{
	Mat tmpImg;
	resize(img, tmpImg, Size(orgImg.cols, orgImg.rows));
	//imshow("Img1", img);
	//imshow("tmpImg", tmpImg);
	//imshow("orgImg", orgImg);
	//HSV颜色特征模型(色调H,饱和度S，亮度V)
	cvtColor(tmpImg, tmpImg, COLOR_BGR2HSV);
	cvtColor(orgImg, orgImg, COLOR_BGR2HSV);
	//直方图尺寸设置
	//一个灰度值可以设定一个bins，256个灰度值就可以设定256个bins
	//对应HSV格式,构建二维直方图
	//每个维度的直方图灰度值划分为256块进行统计，也可以使用其他值
	int hBins = 256, sBins = 256;
	int histSize[] = { hBins,sBins };
	//H:0~180, S:0~255,V:0~255
	//H色调取值范围
	float hRanges[] = { 0,180 };
	//S饱和度取值范围
	float sRanges[] = { 0,255 };
	const float* ranges[] = { hRanges,sRanges };
	int channels[] = { 0,1 };//二维直方图
	MatND hist1, hist2;
	calcHist(&tmpImg, 1, channels, Mat(), hist1,2,histSize, ranges, true, false);
	normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());
	calcHist(&orgImg, 1, channels, Mat(), hist2, 2, histSize, ranges, true, false);
	normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());
	double similarityValue = compareHist(hist1, hist2, CV_COMP_CORREL);
    //cv::imshow("hist1_1", hist1);
    //waitKey(1);
    //cv::imshow("hist2_1", hist2);
    //waitKey(1);
	//cout << "相似度：" << similarityValue << endl;
	if (similarityValue >= 0.85)
	{
		return true;
	}
	return false;
}
 


float Similarity(float* hist1, float* hist2, int length)
{
	float sum = 0;
	float distance;
	for (int i = 0; i < length; i++)
	{
		sum += pow(hist1[i] - hist2[i], 2);
	}	
	distance = sqrt(sum);
	return 1 / (1 + distance);//返回相似度
}


void Rotate(const cv::Mat &srcImage, cv::Mat &dstImage, double angle, cv::Point2f center, double scale)
{
    cv::Mat M = cv::getRotationMatrix2D(center, angle, scale); 
    cv::warpAffine(srcImage, dstImage, M, cv::Size(srcImage.cols, srcImage.rows));  
}




//感知哈希算法相似度
int hashSimilarity(Mat pic1, Mat pic2) {
	Mat matDst1, matDst2;


   //threshold(pic1,matDst1,78, 255,THRESH_BINARY);
   //threshold(pic2,matDst2,78, 255,THRESH_BINARY);

   //threshold(pic1,pic1,78, 255,THRESH_BINARY);
   //threshold(pic2,pic2,78, 255,THRESH_BINARY);


	resize(pic1, matDst1, Size(8,8), 0, 0, INTER_CUBIC);
	resize(pic2, matDst2, Size(8,8), 0, 0, INTER_CUBIC);

	//cv::cvtColor(matDst1, matDst1, CV_BGR2GRAY);
	//cv::cvtColor(matDst2, matDst2, CV_BGR2GRAY);

	cv::Mat temp1 = matDst1;
	cv::Mat temp2 = matDst2;
	cv::cvtColor(temp1 , matDst1, CV_BGR2GRAY);
	cv::cvtColor(temp2 , matDst2, CV_BGR2GRAY);



   imshow("matDst1", matDst1);
   waitKey(1);
   imshow("matDst2", matDst2);
   waitKey(1);

	int iAvg1 = 0, iAvg2 = 0;
	int arr1[64], arr2[64];

	for (int i = 0; i < 8; i++) {
		uchar* data1 = matDst1.ptr<uchar>(i);
		uchar* data2 = matDst2.ptr<uchar>(i);

		int tmp = i * 8;

		for (int j = 0; j < 8; j++) {
			int tmp1 = tmp + j;

			arr1[tmp1] = data1[j] / 4 * 4;
			arr2[tmp1] = data2[j] / 4 * 4;

			iAvg1 += arr1[tmp1];
			iAvg2 += arr2[tmp1];
		}
	}

	iAvg1 /= 64;
	iAvg2 /= 64;

	for (int i = 0; i < 64; i++) {
		arr1[i] = (arr1[i] >= iAvg1) ? 1 : 0;
		arr2[i] = (arr2[i] >= iAvg2) ? 1 : 0;
	}

	int iDiffNum = 0;

	for (int i = 0; i < 64; i++)
		if (arr1[i] != arr2[i])
			++iDiffNum;

	return iDiffNum;
}

int aHash(Mat matSrc1, Mat matSrc2)
{
    Mat matDst1, matDst2;
    cv::resize(matSrc1, matDst1, cv::Size(8, 8), 0, 0, cv::INTER_CUBIC);
    cv::resize(matSrc2, matDst2, cv::Size(8, 8), 0, 0, cv::INTER_CUBIC);

    cv::cvtColor(matDst1, matDst1, CV_BGR2GRAY);
    cv::cvtColor(matDst2, matDst2, CV_BGR2GRAY);

    int iAvg1 = 0, iAvg2 = 0;
    int arr1[64], arr2[64];

    for (int i = 0; i < 8; i++)
    {
        uchar* data1 = matDst1.ptr<uchar>(i);
        uchar* data2 = matDst2.ptr<uchar>(i);

        int tmp = i * 8;

        for (int j = 0; j < 8; j++)
        {
            int tmp1 = tmp + j;

            arr1[tmp1] = data1[j] / 4 * 4;
            arr2[tmp1] = data2[j] / 4 * 4;

            iAvg1 += arr1[tmp1];
            iAvg2 += arr2[tmp1];
        }
    }

    iAvg1 /= 64;
    iAvg2 /= 64;

    for (int i = 0; i < 64; i++)
    {
        arr1[i] = (arr1[i] >= iAvg1) ? 1 : 0;
        arr2[i] = (arr2[i] >= iAvg2) ? 1 : 0;
    }

    int iDiffNum = 0;

    for (int i = 0; i < 64; i++)
        if (arr1[i] != arr2[i])
            ++iDiffNum;

    return iDiffNum;
}


double RotationAngle(int frame_num_1,int frame_num_2,
                     int rect_1_x,int rect_1_y,int rect_1_width,int rect_1_height,
                     int rect_2_x,int rect_2_y,int rect_2_width,int rect_2_height)
{
    double pi=3.141592654;
    struct Point{
        double x;
        double y;
    };

    string path="/home/wl/test_ws/orb_rotate/DetectRotationAngle/";

/*    string s;
    char frame_num[64];
    sprintf(frame_num, "%d", frame_num_1+10000);
    s=frame_num;
    Mat img = imread(path+s.substr(1,4)+".jpg");
    Mat img_1 (img, Rect(rect_1_x, rect_1_y, rect_1_width, rect_1_height));
    sprintf(frame_num, "%d", frame_num_2+10000);
    s=frame_num;
    img = imread(path+s.substr(1,4)+".jpg");
    Mat img_2 (img, Rect(rect_2_x, rect_2_y, rect_2_width, rect_2_height) );
    
    imshow("sd",img_1);
    waitKey();
    imshow("sdd",img_2);
    waitKey();
*/
    
    //Mat img_1=imread("/home/wl/test_ws/orb_rotate/DetectRotationAngle/C0080.png");
    //Mat img_2=imread("/home/wl/test_ws/orb_rotate/DetectRotationAngle/00040.jpg");

    Mat img_1=imread("/home/wl/test_ws/orb_rotate/DetectRotationAngle/C0144.png");
    Mat img_2=imread("/home/wl/test_ws/orb_rotate/DetectRotationAngle/9002.png");
    
    threshold(img_1,img_1,78, 255,THRESH_BINARY);
    threshold(img_2,img_2,78, 255,THRESH_BINARY);
    //imshow("img_2", img_2);
    //waitKey(1);

    // -- Step 1: Detect the keypoints using STAR Detector
    vector<KeyPoint> keypoints_1,keypoints_2;

    Ptr<ORB> orb = ORB::create(9000);
    //Ptr<ORB> orb = ORB::create(10000, 1.1f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

    //orb.detect(img_1, keypoints_1);
    //orb.detect(img_2, keypoints_2);
    
    //if(keypoints_1.size()==0||keypoints_2.size()==0) return 0;
    
    // -- Stpe 2: Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    //orb.compute(img_1, keypoints_1, descriptors_1);
    //orb.compute(img_2, keypoints_2, descriptors_2);

    orb->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1);
    orb->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2);

    Mat showimage1,showimage2;
    drawKeypoints(img_1, keypoints_1, showimage1);
    drawKeypoints(img_2, keypoints_2, showimage2);

    imshow("ORB keypoints", showimage2);
    waitKey(1);


    if(keypoints_1.size()==0||keypoints_2.size()==0) return 0;    
    //-- Step 3: Matching descriptor vectors with a brute force matcher
    BFMatcher matcher(NORM_HAMMING, true);//NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);
    

        cout << "Number of matched points: " << matches.size() << endl;
    
/*        nth_element(matches.begin(),    // initial position
                         matches.begin()+9, // position of the sorted element
                         matches.end());     // end position

        matches.erase(matches.begin()+10, matches.end());
        cout << "Number of new matched points: " << matches.size() << endl;
*/   
    int rand_num_1,rand_num_2;
    Point p1,p2,_p1,_p2;
    double angle_1,angle_2,angle_err;
    vector<double>  angle_err_group(matches.size(),0);
    if(matches.size()>3)
    {
        for(int i=0;i<matches.size();i++)
        {
            rand_num_1=rand()%(matches.size());
            p1.x=keypoints_1[matches[rand_num_1].queryIdx].pt.x;
            p1.y=keypoints_1[matches[rand_num_1].queryIdx].pt.y;
            _p1.x=keypoints_2[matches[rand_num_1].trainIdx].pt.x;
            _p1.y=keypoints_2[matches[rand_num_1].trainIdx].pt.y;
            
            rand_num_2=rand()%(matches.size());
            if(rand_num_2==rand_num_1&&rand_num_1==0) rand_num_2=rand_num_2+1;
            if(rand_num_2==rand_num_1&&rand_num_1==matches.size()-1) rand_num_2=rand_num_2-1;
            
            p2.x=keypoints_1[matches[rand_num_2].queryIdx].pt.x;
            p2.y=keypoints_1[matches[rand_num_2].queryIdx].pt.y;
            _p2.x=keypoints_2[matches[rand_num_2].trainIdx].pt.x;
            _p2.y=keypoints_2[matches[rand_num_2].trainIdx].pt.y;
            
            if((p2.x-p1.x)>=0)
                angle_1=atan((p2.y-p1.y)/(p2.x-p1.x+2.3e-300));
            else
                angle_1=atan((p2.y-p1.y)/(p2.x-p1.x))+pi;
            
            if((_p2.x-_p1.x)>=0)
                angle_2=atan((_p2.y-_p1.y)/(_p2.x-_p1.x+2.3e-300));
            else
                angle_2=atan((_p2.y-_p1.y)/(_p2.x-_p1.x))+pi;
            
            if(angle_1-angle_2>=0)
                angle_err=angle_1-angle_2;
            else
                angle_err=angle_1-angle_2+2*pi;
            
            angle_err_group[i]=angle_err;
            //cout<<p1.y<<","<<p2.y<<"     "<<p1.x<<","<<p2.x<<endl;
            //cout<<"第"<<i<<"个差:"<<angle_err<<endl;
            //angle_sum=angle_sum+angle_err;
        }
        
        //bubble sort angle_err_group
        double temp;
        for(int i = 0;i<matches.size()-1;i++)
        {
            for(int j = 0;j<matches.size()-1-i;j++)
            {
                if(angle_err_group[j] > angle_err_group[j+1])
                {
                    temp=angle_err_group[j];
                    angle_err_group[j]=angle_err_group[j+1];
                    angle_err_group[j+1]=temp;
                }
            }
        }
        
        double err_sum = 0;double count=0;
        for(unsigned long i = matches.size()*4/10;i<=matches.size()*6/10;i++)
        {
            err_sum=err_sum+angle_err_group[i];
            count=count+1;
            //cout<<angle_err_group[i]<<endl;
        }
        
            // -- dwaw matches
            Mat img_mathes;
            drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_mathes);
            // -- show
            imshow("Mathces", img_mathes);
            waitKey(1);
        
        return err_sum/count;
    }
    else
        return 0;
}

Mat rotate(Mat src, double a)
{
    Mat dst;
    Point2f pt(src.cols / 2., src.rows / 2.);
    Mat r = getRotationMatrix2D(pt, a, 1.0);

    warpAffine(src, dst, r, Size(src.cols, src.rows));
    return dst;
}


void similarity(std::string str1, Mat &image)
{
    //Mat img1 = imread( samples::findFile(str1), IMREAD_GRAYSCALE );
    //Mat img2 = imread( samples::findFile(str2), IMREAD_GRAYSCALE );
    Mat img1=imread("/home/wl/test_ws/orb_rotate/DetectRotationAngle/1.jpg");
    Mat img2=imread("/home/wl/test_ws/orb_rotate/DetectRotationAngle/6.jpg");
    //Mat img2=image;
    
    threshold(img1,img1,78, 255,THRESH_BINARY);
    threshold(img2,img2,20, 255,THRESH_BINARY);
    //imshow("sim_img2", img2);
    //waitKey(1);


    if ( img1.empty() || img2.empty() )
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
    }

    Ptr<ORB> detector = ORB::create();
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);

    std::vector< cv::DMatch > firstMatches, secondMatches;
    matcher->match( descriptors1, descriptors2, firstMatches );
    matcher->match( descriptors2, descriptors1, secondMatches );

    int bestMatchesCount = 0;
    std::vector< cv::DMatch > bestMatches;

    for(uint i = 0; i < firstMatches.size(); ++i)
    {
        cv::Point matchedPt1 = keypoints1[i].pt;
        cv::Point matchedPt2 = keypoints2[firstMatches[i].trainIdx].pt;

        bool foundInReverse = false;

        for(uint j = 0; j < secondMatches.size(); ++j)
        {
            cv::Point tmpSecImgKeyPnt = keypoints2[j].pt;
            cv::Point tmpFrstImgKeyPntTrn = keypoints1[secondMatches[j].trainIdx].pt;
            if((tmpSecImgKeyPnt == matchedPt2) && ( tmpFrstImgKeyPntTrn == matchedPt1))
            {
                foundInReverse = true;
                break;
            }
        }
        if(foundInReverse)
        {
            bestMatches.push_back(firstMatches[i]);
            bestMatchesCount++;
        }
    }

    double minKeypoints = keypoints1.size() <= keypoints2.size() ? keypoints1.size() : keypoints2.size();
    double number = ((bestMatchesCount/minKeypoints) * 100);
    int tmpNumber = (int)number;
     std::cout<<"ORB score!!!: "<<number<<std::endl;
    if(tmpNumber >= 50)
        std::cout<<number<<std::endl;
}





float calcOrientationHist(const cv::Mat &img, Point pt, int radius, float *hist, int n, int isSmoothed, int isWeighted, float weighted_sigma)
{
    //radius  should be based on Pt half-centric square side length
        int i, j, k, len = (radius*2+1)*(radius*2+1);
        //Center-weighted using a Gaussian function
        float expf_scale = -1.f/(2.f * weighted_sigma * weighted_sigma);
        //Why add 4 it is to give temporary histogram open extra four storage locations,
        //Used to store temphist [-1], temphist [-2], temphist [n], temphist [n + 1] of
        //Why add n it, this n positions are reserved temphist [0 ... n-1] of
        //Why do len * 4, 4 len array location which is left to the
        //length of the X, Y, W and the direction of the Ori
        cv::AutoBuffer<float> buf(len*4 + n+4);
        //X is the transverse gradient, Y is a longitudinal gradient,
        //Mag is gradient magnitude = sqrt (X ^ 2 + Y ^ 2),
        //Ori is the gradient direction = arctan (Y / X)
        float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
        float* temphist = W + len + 2;//Plus 2 is used to store temphist [-1], temphist [-2]

        //Temporary histogram cleared
        for( i = 0; i < n; i++ )
            temphist[i] = 0.f;

        //Down, left to right scan seek horizontal,
        //vertical gradient values and corresponding weights from the
        for( i = -radius, k = 0; i <= radius; i++ )
        {
            int y = pt.y + i;//The first point of the original image img pt.y + i row
            if( y <= 0 || y >= img.rows - 1 )//Border checks
                continue;
            for( j = -radius; j <= radius; j++ )
            {
                int x = pt.x + j;//The first point of the original image img pt.x + j column
                if( x <= 0 || x >= img.cols - 1 )//Border checks
                    continue;
                //Transverse gradient
                float dx = (float)(img.at<uchar>(y, x+1) - img.at<uchar>(y, x-1));
                //Longitudinal gradient
                float dy = (float)(img.at<uchar>(y-1, x) - img.at<uchar>(y+1, x));
                //Save longitudinal and transverse gradient gradientSave longitudinal and transverse gradient gradient
                X[k] = dx; Y[k] = dy;
                //Calculating a weighted array
                if(isWeighted)
                    W[k] = (i*i + j*j)*expf_scale;
                else
                    W[k] = 1.f; //If you do not weighted, the right point on the weight of each statistic is the same
                k++;
            }
        }
        //Copy the actual statistics point to len, since the rectangular local neighborhood may exceed the image boundary,
        len = k;//So the actual number of points is always less than or equal (radius * 2 + 1) * (radius * 2 + 1)

        //Calculated gradient magnitude at a specified pixel in the neighborhood, and the right to re-gradient direction
        //exp(W, W, len); //Weights
        //fastAtan2(Y, X, Ori, len, true);//Gradient direction
        //magnitude(X, Y, Mag, len);//Gradient magnitude


        //Fill temporary histogram, the horizontal axis is the direction of the gradient angle [0,360), bin width of n / 360;
        //Right vertical axis is multiplied by the corresponding weight gradient magnitude
        for( k = 0; k < len; k++ )
        {
            int bin = cvRound((n/360.f)*Ori[k]);//K-th angle is obtained Ori [k] of bin index number
            if( bin >= n )
                bin -= n;
            if( bin < 0 )
                bin += n;
            temphist[bin] += W[k]*Mag[k];
        }

        if(isSmoothed)
        {
            // Histogram smoothing, smoothing into the output histogram array
            temphist[-1] = temphist[n-1];
            temphist[-2] = temphist[n-2];
            temphist[n] = temphist[0];
            temphist[n+1] = temphist[1];
            for( i = 0; i < n; i++ )
            {
                hist[i] = (temphist[i-2] + temphist[i+2])*(1.f/16.f) +
                    (temphist[i-1] + temphist[i+1])*(4.f/16.f) +
                    temphist[i]*(6.f/16.f);
            }
        }
        else  //Not smooth histogram
        {
            for( i = 0; i < n; i++ )
            {
                hist[i] = temphist[i];
            }
        }

        //Maximum gradient histogram
        float maxval = hist[0];
        for( i = 1; i < n; i++ )
            maxval = std::max(maxval, hist[i]);

        //cal direction uncertain
        float sum=0,certain;
        for(int i = 0;i<n;i++)
        {
            if(hist[i]==maxval)
            {

            }else{
                sum+=hist[i];
            }

        }
        certain = maxval-sum/(n-1);

        //return certain;
        return maxval;
}

void DrawHist(Mat& hist, string&  winname)
{
    Mat drawHist;

    int histSize = hist.rows;
    // 创建直方图画布
   int hist_w = 360; int hist_h = 360;//直方图图像的宽度和高度
   int bin_w = cvRound( (double) hist_w/histSize );//直方图中一个矩形条纹的宽度
   Mat histImage( hist_w, hist_h, CV_8UC3, Scalar( 0,0,0) );//创建画布图像

   /// 将直方图归一化到范围 [ 0, histImage.rows ]
   normalize(hist, drawHist, 0,histImage.rows, NORM_MINMAX, -1, Mat() );

   /// 在直方图画布上画出直方图
   //for(int i=1;i(i-1))),Scalar(0,0,255),1,8,0){  
       //折线表示 
      /* line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at(i-1)) ) ,
                      Point( bin_w*(i), hist_h - cvRound(hist.at(i)) ),
                      Scalar( 0, 0, 255), 1, 8, 0  );*/

   // }
    /// 显示直方图

}




	void imageCallback(const sensor_msgs::Image::ConstPtr& image_msg){
	    cv::Mat color_mat(image_msg->height,image_msg->width,CV_MAKETYPE(CV_8U,3),const_cast<uchar*>(&image_msg->data[0]), image_msg->step);
	    //cv::cvtColor(color_mat,color_mat,cv::COLOR_BGR2RGB);
	    width  = (size_t)color_mat.size().width;
            height = (size_t)color_mat.size().height;
            //ROS_INFO("image_width : %d %d", width,height);
/*	    frame_to_blob(color_mat, inferReq, inputLayerName);
	    start = ros::Time::now().toSec(); //Latency timing

	    inferReq->Infer(); //Inference starts
*/


		//std::string file_xml = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/best-sim.xml";
		//std::string file_bin = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/best-sim.bin";
/*		std::string file_xml = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/best-tiny.xml";
		std::string file_bin = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/best-tiny.bin";
                //std::string file_xml = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/rotate_ref_best.xml";
		//std::string file_bin = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/rotate_ref_best.bin";

		YOLOObjectDetection yolo_detector;
		std::vector<YOLOObjectDetection::Detection> outputs;
		yolo_detector.detect(file_xml, file_bin, color_mat, 0, outputs);

	        float scale_x = width / 640.0;
	        float scale_y = height / 640.0;

	        //yolo_detector.inferAndOutput(color_mat, infer_request, input_info, output_info, scale_x, scale_y);


		int detections = outputs.size();
		//ROS_INFO("detections: %d", detections);
		for (int i = 0; i < detections; ++i)
		{

			auto detection = outputs[i];
			auto confidences = detection.confidence;
			auto box = detection.box;
			auto classId = detection.class_id;
			vector<std::string> m_classNames;
			const vector<Scalar> colors = { Scalar(255, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 0) };
			const auto color = colors[classId % colors.size()];
			rectangle(color_mat, box, color, 3);

                        //ROS_INFO("Confidence: %d %f %d %d %d %d \n", classId, confidences, box.x, box.y, box.width, box.height);
			//rectangle(color_mat, Point(box.x, box.y - 40), Point(box.x + box.width, box.y), color, FILLED);

			//putText(color_mat, m_classNames[classId].c_str(), Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 0), 2);
                        
                        yolo_center.x = box.x + box.width/2.0;
                        yolo_center.y = box.y + box.height/2.0;

                        object_pose_.pose.position.z = 0.13 * fx_color / box.width;
                        object_pose_.pose.position.x = object_pose_.pose.position.z * (yolo_center.x-cx_color)/fx_color;
                        object_pose_.pose.position.y = object_pose_.pose.position.z * (yolo_center.y-cy_color)/fy_color;
                        
                        object_pose_.pose.orientation.w = 1;
                        object_pose_.pose.orientation.z = 0;
                        object_pose_.pose.orientation.y = 0;
                        object_pose_.pose.orientation.x = 0;
                        //ROS_INFO("object_pose_: %f %f %f \n", object_pose_.pose.position.x, object_pose_.pose.position.y, object_pose_.pose.position.z);


		}

		cv::imshow("YOLO", color_mat);
		waitKey(1);
*/


 //auto start = chrono::high_resolution_clock::now();


                //cout<<RotationAngle(1,4,378,316,77,114,375,318,75,112)*180/3.141592654<<endl;

                //cout<<evalAngle(  cv::imread("/home/wl/test_ws/orb_rotate/DetectRotationAngle/C0080.png",cv::IMREAD_GRAYSCALE) ,
                             //cv::imread("/home/wl/test_ws/orb_rotate/DetectRotationAngle/0007.png",cv::IMREAD_GRAYSCALE)     )<<endl;


    //auto end = chrono::high_resolution_clock::now();
    //std::chrono::duration<double> diff = end - start;
    //cout<<"RotationAngle use "<<diff.count()<<" s" << endl;

                Mat test_img = imread("/home/wl/realsense_d435/cam/C0142/A0023.jpg"); //00296 00164 00341 00287


    //Mat dst;
    //dst = rotate(test_img, 26);
    //cv::imshow("dst", dst);
    //waitKey(1);
    similarity("a", test_img);


                auto start = chrono::high_resolution_clock::now();

                vector<Rotate_Detector::Object> detected_objects;
                rotate_detector.process_frame(test_img, detected_objects); //color_mat

                auto end = chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = end - start;
                cout<<"detect use "<<diff.count()<<" s" << endl;

	        float scale_x = width / 640.0;
	        float scale_y = height / 640.0;


		int detections = detected_objects.size();
		//ROS_INFO("detections11: %d", detections);

		for (int i = 0; i < detections; ++i)
		{

			auto detection = detected_objects[i];
			auto confidences = detection.prob;
			auto box = detection.rect;
			auto classId = detection.classId;
                        auto angle = detection.angle;

                        yolo_center.x = box.x + box.width/2.0;
                        yolo_center.y = box.y + box.height/2.0;

			vector<std::string> m_classNames;
			const vector<Scalar> colors = { Scalar(255, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 0) };
			const auto color = colors[classId % colors.size()];

                        cv::Point2f center(yolo_center.x, yolo_center.y);
	                Rotate(test_img, test_img, angle, center, 1.0);

			rectangle(test_img, box, color, 3);

                        ROS_INFO("Confidence11: %d %f %d %d %d %d  %f\n", classId, confidences, box.x, box.y, box.width, box.height, angle);
			//rectangle(color_mat, Point(box.x, box.y - 40), Point(box.x + box.width, box.y), color, FILLED);

			//putText(color_mat, m_classNames[classId].c_str(), Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 0), 2);
                        

                        object_pose_.pose.position.z = 0.13 * fx_color / box.width;
                        object_pose_.pose.position.x = object_pose_.pose.position.z * (yolo_center.x-cx_color)/fx_color;
                        object_pose_.pose.position.y = object_pose_.pose.position.z * (yolo_center.y-cy_color)/fy_color;
                        
                        object_pose_.pose.orientation.w = 1;
                        object_pose_.pose.orientation.z = 0;
                        object_pose_.pose.orientation.y = 0;
                        object_pose_.pose.orientation.x = 0;
                        //ROS_INFO("object_pose_: %f %f %f \n", object_pose_.pose.position.x, object_pose_.pose.position.y, object_pose_.pose.position.z); 


		}

		cv::imshow("YOLO", test_img);
		waitKey(1);


    //yoloOneOutput yolovvino; //buxing zhan cpu gao

/*    Mat src = color_mat;
    Mat src2 = src.clone();
    int width = src.cols;
    int height = src.rows;
    int channel = src.channels();
    double scale = min(640.0 / width, 640.0 / height);
    int w = round(width * scale);
    int h = round(height * scale);
    //cout << "w: " << w << endl;
    //cout << "h: " << h << endl;
    Mat src3;
    resize(src2, src3, Size(w, h));
    int top = 0, bottom = 0, left = 0, right = 0;
    if (w > h)
    {
        top = (w - h) / 2;
        bottom = (w - h) - top;
    }
    else if (h > w)
    {
        left = (h - w) / 2;
        right = (h - w) - left;
    }
    copyMakeBorder(src3, src3, top, bottom, left, right, BORDER_CONSTANT, Scalar(114,114,114));

    vector<yoloOneOutput::Object> detected_objects;
    auto start = chrono::high_resolution_clock::now();
    //yolovvino.process_frame(color_mat,detected_objects);
    yolovvino.process_frame(src3,detected_objects);
    auto end = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    //cout<<"use "<<diff.count()<<" s" << endl;
    for(size_t i=0;i<detected_objects.size();++i){
        //int xmin = detected_objects[i].rect.x;
        //int ymin = detected_objects[i].rect.y;
        int xmin = max(detected_objects[i].rect.x - left, 0);
        int ymin = max(detected_objects[i].rect.y - top, 0);
        int width = detected_objects[i].rect.width;
        int height = detected_objects[i].rect.height;

        //Rect rect(xmin, ymin, width, height);
        Rect rect(int(xmin / scale), int(ymin / scale), int(width / scale), int(height / scale));
        auto confidences = detected_objects[i].prob;
        auto box = detected_objects[i].rect;
	auto classId = detected_objects[i].id;

        ROS_INFO("Confidence: %d %f %d %d %d %d \n", classId, confidences, box.x, box.y, box.width, box.height);
        const vector<Scalar> colors = { Scalar(255, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 0) };
	const auto color = colors[classId % colors.size()];

        //rectangle(color_mat, box, color, 3);
        rectangle(src2, rect, Scalar(0, 0, 255), 1, LINE_8, 0);

    }
  
    imshow("result",src2);
    waitKey(1);
*/


//	        YOLOVINO yolov7vino;
/*	        int frameCount = 0;
	        int fps = 0;

                //auto start = chrono::high_resolution_clock::now();

		std::vector<YOLOVINO::Detection> yolo_outputs;
		yolov7vino.detect(color_mat, yolo_outputs);

		//yolov7vino.drawRect(color_mat, yolo_outputs);

                //auto end = chrono::high_resolution_clock::now();
                //std::chrono::duration<double> diff = end - start;
                //cout<<"use "<<diff.count()<<" s" << endl;

                int detections = yolo_outputs.size();
                for (int i = 0; i < detections; ++i)
                {


		    vector<std::string> m_classNames;
		    const vector<Scalar> colors = { Scalar(255, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 0) };
                    auto detection = yolo_outputs[i];
	            auto confidences = detection.confidence;
                    auto box = detection.box;
                    auto classId = detection.class_id;
                    const auto color = colors[classId % colors.size()];
                    rectangle(color_mat, box, color, 3);
 
                    //rectangle(color_mat, Point(box.x, box.y - 40), Point(box.x + box.width, box.y), color, FILLED);
                    //putText(color_mat, m_classNames[classId].c_str(), Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 0), 2);
                   //ROS_INFO("Confidence: %d %f %d %d %d %d \n", classId, confidences, box.x, box.y, box.width, box.height);
                    ROS_INFO("Confidence: %d %f %d %d %d %d \n", classId, confidences, box.x, box.y, box.width, box.height);

                        yolo_center.x = box.x + box.width/2.0;
                        yolo_center.y = box.y + box.height/2.0;

                        object_pose_.pose.position.z = 0.13 * fx_color / box.width;
                        object_pose_.pose.position.x = object_pose_.pose.position.z * (yolo_center.x-cx_color)/fx_color;
                        object_pose_.pose.position.y = object_pose_.pose.position.z * (yolo_center.y-cy_color)/fy_color;
                        
                        object_pose_.pose.orientation.w = 1;
                        object_pose_.pose.orientation.z = 0;
                        object_pose_.pose.orientation.y = 0;
                        object_pose_.pose.orientation.x = 0;
                        ROS_INFO("object_pose_: %f %f %f \n", object_pose_.pose.position.x, object_pose_.pose.position.y, object_pose_.pose.position.z);

                }

		//cv::namedWindow("YOLO", cv::WINDOW_NORMAL);
		cv::imshow("YOLO", color_mat);
                waitKey(1);
*/



/*               //YOLO onnxruntime_detector;
               YOLO onnxruntime_detector;
ROS_ERROR("1111111111111");
	       //onnxruntime_detector.detect(color_mat);
	       string imgpath = "/home/wl/test_ws/swarm_ws/yolo_rot_detection/rotate-yolov5-opencv-onnxrun-main/onnxruntime/images/P0032.png";
	       Mat srcimg = imread(imgpath);
	       int64 start = getTickCount();
               onnxruntime_detector.detect(srcimg);
               float request_end_time = (getTickCount() - start) / getTickFrequency();
               //printf("request_end: %f \n", request_end_time);

	       //static const string kWinName = "Deep learning object detection in ONNXRuntime";
	       //namedWindow(kWinName, WINDOW_NORMAL);
	       imshow("YOLO", color_mat);
	       waitKey(0);
*/


/*
	    if (OK == inferReq->Wait(IInferRequest::WaitMode::RESULT_READY)) {
			//Take Latency calculation
			stop = ros::Time::now().toSec();
			double duration = stop - start; 
			count++;
			latency_counter++;

			//Calculate running average latency
			if(latency_counter%4==0){
			    running_avg_latency += duration;
			    running_avg_latency = (float) running_avg_latency / 4;
			    std::cout<<"- Running Average Latency = "<< running_avg_latency << " secs -\n";
			    running_avg_latency = 0;
			} else {
			    running_avg_latency += duration;
			}
			
			try{
			    //Fetch the results associated with the inference request
				compute_results = inferReq->GetBlob(outputLayerName)->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
			}
			catch(const std::exception& e){
				ROS_ERROR("%s",e.what());
				std::cout<<"Error retrieving inference results. More:" << e.what() << std::endl;
			}

	    	for (int i = 0; i < results_number; i++) {
				//Extract the individual result values from the aggregated result
	    	    float result_id = compute_results[i * object_size + 0];
	            int result_label = static_cast<int>(compute_results[i * object_size + 1]);
	    	    float result_confidence = compute_results[i * object_size + 2];
	    	    float result_xmin = compute_results[i * object_size + 3];
	    	    float result_ymin = compute_results[i * object_size + 4];
	    	    float result_xmax = compute_results[i * object_size + 5];
	    	    float result_ymax = compute_results[i * object_size + 6];
				//Print out the results
				if(result_label && result_confidence > confidence_thresh){
					printf("------DETECTION------\nLabel = %d\nConfidence = %.1f\nXmin = %.1f\nYmin = %.1f\nWidth = %.1f\nHeight = %.1f\n---------------------", result_label, result_confidence*100, result_xmin*width, result_ymin*height, (result_xmax-result_xmin)*width, (result_ymax-result_ymin)*height);
				}

				if (result_confidence > confidence_thresh){
					//Load the results to message object
					result_obj.label= result_label;
					result_obj.confidence=result_confidence;
					result_obj.x=result_xmin;
					result_obj.y=result_ymin;
					result_obj.width=result_xmax-result_xmin;
					result_obj.height=result_ymax-result_ymin;
					results.objects.push_back(result_obj);

					//Publish the results obtained/
					try{
						std::cout<<"\nPublishing result\n";
						results.header.stamp=ros::Time::now();
						det_results.publish(results);
						results.objects.clear();
					}
					catch(const std::exception& e){
						ROS_ERROR("%s",e.what());
						std::cout<<"Error publishing inference result. More:" << e.what() << std::endl;
					}

					//Calculate inferences per second
					double stop_time = ros::Time::now().toSec();
					fps = (stop_time - start_time > 1)?count:fps;
					if(stop_time - start_time > 1){
						std::cout<<"\n- Inferences per second = "<< fps << " -\n";
						start_time = ros::Time::now().toSec();
						count = 0;
					}


                                        std::vector<std::string> labels;
                                        labels.reserve(3);

                                        labels.push_back("no-mark");
                                        labels.push_back("mark");
                                        labels.push_back("none");

                                        std::ostringstream conf;
                                        conf << ":" << std::fixed << std::setprecision(3) << result_confidence;
                                        int color[3][3] ={{0,0,255},{0,255,0},{255,0,0}};

                                        //cv::putText(color_mat, (!labels.empty() ? labels[result_label] : std::string("label #") + std::to_string(result_label)) + conf.str(), cv::Point2f(result_xmin, result_ymin - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255));

                                        //cv::rectangle(color_mat, cv::Point2f(result_xmin, result_ymin), cv::Point2f(result_xmax, result_ymax), cv::Scalar(color[int(result_label)][0],color[int(result_label)][1],color[int(result_label)][2]));

                                //cv::imshow("Detection results", color_mat);
                                //cv::waitKey(1);
				}
			}

                        //cv::imshow("Detection results", color_mat);
                        //cv::waitKey(1);
    
		}
*/ 		//Callback Code Ends Here//
	}

	private:
  	ros::NodeHandle nh; 
  	ros::Publisher det_results;
  	ros::Subscriber image_sub;
	InferRequest::Ptr inferReq;            //For holding the inference request
	ExecutableNetwork model_network;       //For loading network to device
	std::string inputLayerName;        
	std::string outputLayerName;
	size_t width, height;
	float *compute_results;                //For fetching results into
	rosvino::Object result_obj;            //output msg classes
	rosvino::Objects results;
	SizeVector output_dimension;           //To hold information about output
	int results_number;
	int object_size;

//InferRequest infer_request;
//InputsDataMap input_info;
//OutputsDataMap output_info;

        geometry_msgs::PoseStamped object_pose_;

};



int main(int argc, char **argv) {
	ros::init(argc, argv, "detect");
	ros::NodeHandle nh;
	Core ie;

	if(!nh.getParam("/detect/threshold", confidence_thresh)){
		confidence_thresh=0.92;
	}
	if(!nh.getParam("/detect/target_device", device)){
		device="GPU";
	}
	if(!nh.getParam("/detect/network", network_loc)){
		//network_loc = "/home/wl/test_ws/darknet/src/rm_project/openvino/pin_defects_model/pin_defects_detector.xml";
		//network_loc = "/home/wl/test_ws/darknet/src/rm_project/openvino/color_model/color.xml";
		//network_loc = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/best.xml";
		network_loc = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/best-tiny.xml";


	}



	//CNNNetwork network_reader = ie.ReadNetwork(network_loc);
	Detect detect(nh);
	ros::spin();
}

