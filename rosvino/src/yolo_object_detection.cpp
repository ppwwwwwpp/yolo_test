#include <iostream>
#include "yolo_object_detection.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace InferenceEngine;

std::vector<float> anchors = {
	10,13, 16,30, 33,23,
	30,61, 62,45, 59,119,
	116,90, 156,198, 373,326

//        12,16, 19,36, 40,28,
//        36,75, 76,55, 72,146,
//        142,110, 192,243, 459,401

};

int get_anchor_index(int scale_w, int scale_h) {
	if (scale_w == 20) {
		return 12;
	}
	if (scale_w == 40) {
		return 6;
	}
	if (scale_w == 80) {
		return 0;
	}
	return -1;
}

float get_stride(int scale_w, int scale_h) {
	if (scale_w == 20) {
		return 32.0;
	}
	if (scale_w == 40) {
		return 16.0;
	}
	if (scale_w == 80) {
		return 8.0;
	}
	return -1;
}

float sigmoid_function(float a)
{
	float b = 1. / (1. + exp(-a));
	return b;
}

void YOLOObjectDetection::detect(std::string xml, std::string bin, cv::Mat color_mat, int camera_index, vector<Detection> &net_outputs) {
	VideoCapture cap;
	Mat frame;
	/*if (camera_index == 0) {
		cap.open(0);
	}
	if (camera_index == 1) {
		cap.open(filePath);
	}
	if (camera_index == -1) {
		frame = imread(filePath);
	}
	if (frame.empty()) {
		cap.read(frame);
	}
*/
        frame = color_mat;
	int image_height = frame.rows;
	int image_width = frame.cols;

	// 创建IE插件, 查询支持硬件设备 480 640
	Core ie;
//	vector<string> availableDevices = ie.GetAvailableDevices();
//	for (int i = 0; i < availableDevices.size(); i++) {
//		printf("supported device name : %s \n", availableDevices[i].c_str());
//	}

	//  加载检测模型
	auto network = ie.ReadNetwork(xml, bin);
	// auto network = ie.ReadNetwork(xml);
	//cout << "network layer count: " << network.layerCount() << endl;
	// 请求网络输入与输出信息
	InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
	InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());

	// 设置输入格式
	for (auto &item : input_info) {
		auto input_data = item.second;
		input_data->setPrecision(Precision::FP32);
		input_data->setLayout(Layout::NCHW);
		input_data->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
		input_data->getPreProcess().setColorFormat(ColorFormat::RGB);
	}

	// 设置输出格式
	for (auto &item : output_info) {
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP32);

                m_outputName = item.first;
//                std::cout << "output name = " << m_outputName << std::endl;
	}

	auto executable_network = ie.LoadNetwork(network, "CPU");

        //auto device_name = "GPU";
        //executable_network = ie.compile_model(network, device_name);
	// 请求推断图
	auto infer_request = executable_network.CreateInferRequest();
	float scale_x = image_width / 640.0;
	float scale_y = image_height / 640.0;

/*
	if (camera_index == -1) {
		inferAndOutput(frame, infer_request, input_info, output_info, scale_x, scale_y);
		cv::imshow("YOLO", frame);
	}
	else {
		while (true) {
			bool ret = cap.read(frame);
			if (frame.empty()) {
				break;
			}
			inferAndOutput(frame, infer_request, input_info, output_info, scale_x, scale_y);
			cv::imshow("YOLO", frame);
			char c = cv::waitKey(1);
			if (c == 27) {
				break;
			}
		}
	}
*/
	inferAndOutput(frame, infer_request, input_info, output_info, scale_x, scale_y);

        net_outputs = outputs;

//	cv::imshow("YOLO", frame);
//	waitKey(1);
//	destroyAllWindows();
}

void YOLOObjectDetection::inferAndOutput(cv::Mat &frame, InferenceEngine::InferRequest &infer_request, 
	InferenceEngine::InputsDataMap & input_info, InferenceEngine::OutputsDataMap &output_info, float sx, float sy) {
	int64 start = getTickCount();

	// 处理解析输出结果
	vector<Rect> boxes;
	vector<int> classIds;
	vector<float> confidences;

	/** Iterating over all input blobs **/
	for (auto & item : input_info) {
		auto input_name = item.first;

		/** Getting input blob **/
		auto input = infer_request.GetBlob(input_name);
		size_t num_channels = input->getTensorDesc().getDims()[1];
		size_t h = input->getTensorDesc().getDims()[2];
		size_t w = input->getTensorDesc().getDims()[3];
		size_t image_size = h*w;
		Mat blob_image;
		resize(frame, blob_image, Size(w, h));
		cvtColor(blob_image, blob_image, COLOR_BGR2RGB);
                //printf("num_channelsqqqq!!!:  %d %d %d %d \n", num_channels, h, w, image_size);
		// NCHW
		float* data = static_cast<float*>(input->buffer());
		for (size_t row = 0; row < h; row++) {
			for (size_t col = 0; col < w; col++) {
				for (size_t ch = 0; ch < num_channels; ch++) {
					data[image_size*ch + row*w + col] = float(blob_image.at<Vec3b>(row, col)[ch]) / 255.0;
				}
			}
		}
	}

        //float request_start = getTickFrequency() / (getTickCount() - start);
        float request_start_time = (getTickCount() - start) / getTickFrequency();
        //printf("request_start: %f \n", request_start_time);
	// 执行预测
	infer_request.Infer();

	//float request_end = getTickFrequency() / (getTickCount() - start);
        float request_end_time = (getTickCount() - start) / getTickFrequency();
        printf("request_end: %f \n", request_end_time);

	for (auto &item : output_info) {
		auto output_name = item.first;
		auto output = infer_request.GetBlob(output_name);

		const float* output_blob = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output->buffer());
		
		const SizeVector outputDims = output->getTensorDesc().getDims();
		const int out_n = outputDims[0];
		const int out_c = outputDims[1];
		const int side_h = outputDims[2];
		const int side_w = outputDims[3];
		const int side_data = outputDims[4];
                //printf("side_hqqqq!!!: %d %d %d %d %d \n", out_n, out_c, side_h, side_w, side_data); //1 3 80 80 7    1 3 40 40 7    1 3 20 20 7
		float stride = get_stride(side_h, side_h);
		int anchor_index = get_anchor_index(side_h, side_h);
		int side_square = side_h*side_w;
		int side_data_square = side_square*side_data;
		int side_data_w = side_w*side_data;

		for (int i = 0; i < side_square; ++i) {
			for (int c = 0; c < out_c; c++) {
				int row = i / side_h;
				int col = i % side_h;
				int object_index = c*side_data_square + row*side_data_w + col*side_data;

				// 阈值过滤
				float conf = sigmoid_function(output_blob[object_index + 4]);
				if (conf < 0.25) {
					continue;
				}

				// 解析cx, cy, width, height
				float x = (sigmoid_function(output_blob[object_index]) * 2 - 0.5 + col)*stride;
				float y = (sigmoid_function(output_blob[object_index + 1]) * 2 - 0.5 + row)*stride;
				float w = pow(sigmoid_function(output_blob[object_index + 2]) * 2, 2)*anchors[anchor_index + c * 2];
				float h = pow(sigmoid_function(output_blob[object_index + 3]) * 2, 2)*anchors[anchor_index + c * 2 + 1];
				float max_prob = -1;
				int class_index = -1;
                                //printf("side_hqqqq!!!: %f %f %f %f \n", x, y, w, h);
				// 解析类别输出的网络宽度是类别数+5
				//for (int d = 5; d < 85; d++) {
				for (int d = 5; d < 7; d++) {
					float prob = sigmoid_function(output_blob[object_index + d]);
					if (prob > max_prob) {
						max_prob = prob;
						class_index = d - 5;
					}
				}

				// 转换为top-left, bottom-right坐标
				int x1 = saturate_cast<int>((x - w / 2) * sx);  // top left x
				int y1 = saturate_cast<int>((y - h / 2) * sy);  // top left y
				int x2 = saturate_cast<int>((x + w / 2) * sx);  // bottom right x
				int y2 = saturate_cast<int>((y + h / 2) * sy); // bottom right y
                                //printf("top-left,bottom-right!!!: %d %d %d %d \n", x1, y1, x2, y2);
				// 解析输出

				classIds.push_back(class_index);
				confidences.push_back((float)conf);
				boxes.push_back(Rect(x1, y1, x2 - x1, y2 - y1));

                                //printf("Confidence: %d %d %f %f %f %f\n", classIds.size(), classIds[0], x, y, w, h);
                                //printf("------DETECTION------\nLabel = %d\nConfidence = %.1f\nXmin = %.1f\nYmin = %.1f\nWidth = %.1f\nHeight = %.1f\n---------------------", "aa", confidences, x1, y1, x2 - x1, y2 - y1);
			}
		}
	}

	vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.5, indices);
        //std::cout << "classIds = " << classIds << std::endl;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		rectangle(frame, box, Scalar(140, 199, 0), 4, 8, 0);


                Detection result;
                result.class_id = classIds[idx];
                result.confidence = confidences[idx];
                result.box = boxes[idx];
		outputs.push_back(result);
                printf("Confidence: %d %d %f \n", idx, classIds[idx], confidences[idx]);
                //printf("box!! : %d %d %d %d \n", box.x, box.y, box.width, box.height);
                
	}
	float fps = getTickFrequency() / (getTickCount() - start);
	float time = (getTickCount() - start) / getTickFrequency();

	ostringstream ss;
	ss << "FPS : " << fps << " detection time: " << time * 1000 << " ms";
	cv::putText(frame, ss.str(), Point(20, 50), 0, 1.0, Scalar(0, 0, 255), 2);
}
