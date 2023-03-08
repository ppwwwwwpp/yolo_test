#include "rotate_detector.h"

Rotate_Detector::Rotate_Detector() {

    //string xml_path = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/rotate_ref_best.xml";
    //string xml_path = "/home/wl/test_ws/swarm_ws/src/rosvino/models/rotate_ref_best_n.xml";
    string xml_path = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/best_s_6.xml";
    init(xml_path, 0.7, 0.5);  

}

Rotate_Detector::~Rotate_Detector() {}


bool Rotate_Detector::parse_yolov5(const Blob::Ptr& blob, int net_grid, float cof_threshold, vector<Rect>& o_rect, vector<float>& o_rect_cof, vector<int>& classIds, vector<float>& rotate_angle)
{
    vector<int>              anchors     = get_anchors(net_grid);
    LockedMemory<const void> blobMapped  = as<MemoryBlob>(blob)->rmap();
    const float*             output_blob = blobMapped.as<float*>();

    float x_factor = float(col / 640.0);
    float y_factor = float(1920.0 / 640.0);


    //cout << "w: " << anchors << endl;
    //printf("output_blob: %d \n", blobMapped.as<float*>());
    // int item_size = 6;
    //int    item_size = 187;
    int    item_size = 191;
    size_t anchor_n  = 3;

    for (int n = 0; n < anchor_n; ++n)
        for (int i = 0; i < net_grid; ++i)
            for (int j = 0; j < net_grid; ++j) {
                double box_prob = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 4];
		//double box_prob = output_blob[n*net_grid*net_grid*item_size + i * net_grid + j + 4 * net_grid*net_grid];


                box_prob        = sigmoid(box_prob);

                //cout << "box_prob: " << box_prob << endl;

                //框置信度不满足则整体置信度不满足
                if (box_prob < cof_threshold) continue;

                //注意此处输出为中心点坐标,需要转化为角点坐标
                double x = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 0];
                double y = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 1];
                double w = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 2];
                double h = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 3];

                double max_prob = 0;
                int    idx      = 0;
                //for (int t = 5; t < 7; ++t) {
                for (int t = 5; t < 11; ++t) {
                    double tp = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + t];
                    //cout << "tp: " << tp << endl;
                    tp        = sigmoid(tp);
                    if (tp > max_prob) {
                        max_prob = tp;
                        idx      = t;
                    }
                }

                double max_prob_angle = 0;
                int    angle      = 0;
                //for (int t = 7; t < 187; ++t) {
                for (int t = 11; t < 191; ++t) {
                    double tp_angle = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + t];
                    tp_angle    = sigmoid(tp_angle);


                    if (tp_angle > max_prob_angle) {
                        max_prob_angle = tp_angle;
                        angle      = t;
                    }

                }
                //cout << "angle: " << angle << endl;


                float cof = box_prob * max_prob;
                //对于边框置信度小于阈值的边框,不关心其他数值,不进行计算减少计算量
                if (cof < cof_threshold) continue;

                x = (sigmoid(x) * 2 - 0.5 + j) * 640.0f / net_grid * x_factor;
                y = (sigmoid(y) * 2 - 0.5 + i) * 640.0f / net_grid * y_factor;
                w = pow(sigmoid(w) * 2, 2) * anchors[n * 2] * x_factor;
                h = pow(sigmoid(h) * 2, 2) * anchors[n * 2 + 1] * y_factor;

                cout << "  " << x << "  "<< y <<  "  " << w << "  " << h  << endl;

                double r_x  = x - w / 2;
                double r_y  = y - h / 2;
                Rect   rect = Rect(round(r_x) , round(r_y), round(w), round(h));
                o_rect.push_back(rect);
                o_rect_cof.push_back(cof);
                classIds.push_back(idx - 5);
                //rotate_angle.push_back(angle - 7.0);
                rotate_angle.push_back(angle - 11.0);
            }
    if (o_rect.size() == 0)
        return false;
    else
        return true;
}


bool Rotate_Detector::init(string xml_path, double cof_threshold, double nms_area_threshold)
{
    _xml_path           = xml_path;
    _cof_threshold      = cof_threshold;
    _nms_area_threshold = nms_area_threshold;
    Core ie;
    auto cnnNetwork = ie.ReadNetwork(_xml_path);
    //输入设置
    InputsDataMap   inputInfo(cnnNetwork.getInputsInfo());
    InputInfo::Ptr& input = inputInfo.begin()->second;
    _input_name           = inputInfo.begin()->first;
    input->setPrecision(Precision::FP32);
    input->getInputData()->setLayout(Layout::NCHW);
    ICNNNetwork::InputShapes inputShapes  = cnnNetwork.getInputShapes();
    SizeVector&              inSizeVector = inputShapes.begin()->second;
    cnnNetwork.reshape(inputShapes);
    //输出设置
    _outputinfo = OutputsDataMap(cnnNetwork.getOutputsInfo());
    for (auto& output : _outputinfo) {
        output.second->setPrecision(Precision::FP32);
    }
    //获取可执行网络
    //_network = ie.LoadNetwork(cnnNetwork, "GPU");
     _network = ie.LoadNetwork(cnnNetwork, "CPU");
    return true;
}

//释放资源
bool Rotate_Detector::uninit()
{
    return true;
}

//处理图像获取结果
bool Rotate_Detector::process_frame(Mat& inframe, vector<Object>& detected_objects)
{
    if (inframe.empty()) {
        cout << "无效图片输入" << endl;
        return false;
    }
    //resize(inframe, inframe, Size(640, 640));


	col = inframe.cols;
	row = inframe.rows;
	int max = MAX(col, row);
    Mat result = Mat::zeros(max, max, CV_8UC3);
    inframe.copyTo(result(Rect(0, 0, col, row)));

    cv::Mat blob_image;
    resize(result, blob_image, cv::Size(640, 640));
    cvtColor(blob_image, blob_image, COLOR_BGR2RGB);


    //cvtColor(inframe, inframe, COLOR_BGR2RGB);
    size_t                              img_size      = 640 * 640;
    InferRequest::Ptr                   infer_request = _network.CreateInferRequestPtr();
    Blob::Ptr                           frameBlob     = infer_request->GetBlob(_input_name);
    InferenceEngine::LockedMemory<void> blobMapped    = InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();
    float*                              blob_data     = blobMapped.as<float*>();
    // nchw
    for (size_t row = 0; row < 640; row++) {
        for (size_t col = 0; col < 640; col++) {
            for (size_t ch = 0; ch < 3; ch++) {
                blob_data[img_size * ch + row * 640 + col] = float(blob_image.at<Vec3b>(row, col)[ch]) / 255.0f;
            }
        }
    }
    //执行预测
    auto start = getTickCount();

    infer_request->Infer();

    auto end = getTickCount();
    auto time = (getTickCount() - start) / getTickFrequency();
    cout << "time = " << time << endl;
    //获取各层结果
    vector<Rect>  origin_rect;
    vector<float> origin_rect_cof;
    vector<int> classIds;
    vector<float> rotate_angle;

    int           s[3] = { 80, 40, 20 };
    int           i    = 0;
    for (auto& output : _outputinfo) {
        auto      output_name = output.first;
        Blob::Ptr blob        = infer_request->GetBlob(output_name);
        parse_yolov5(blob, s[i], _cof_threshold, origin_rect, origin_rect_cof, classIds, rotate_angle);
        ++i;
    }



/*
	vector<Rect> rect;
	vector<int> classId;
	vector<float> prob;
	for (auto &item : _outputinfo) {

                int width  = (size_t)inframe.size().width;
                int height = (size_t)inframe.size().height;
                float sx = width / 640.0;
	        float sy = height / 640.0;

		auto output_name = item.first;
		auto output = infer_request->GetBlob(output_name);

		const float* output_blob = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output->buffer());
		
		const SizeVector outputDims = output->getTensorDesc().getDims();
		const int out_n = outputDims[0];
		const int out_c = outputDims[1];
		const int side_h = outputDims[2];
		const int side_w = outputDims[3];
		const int side_data = outputDims[4];
                //printf("side_hqqqq!!!: %d %d %d %d %d \n", out_n, out_c, side_h, side_w, side_data); //1 3 80 80 187    1 3 40 40 187    1 3 20 20 187
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
                                //cout << "conf: " << conf << endl;
				if (conf < 0.5) {
					continue;
				}

				// 解析cx, cy, width, height
				float x = (sigmoid_function(output_blob[object_index]) * 2 - 0.5 + col)*stride;
				float y = (sigmoid_function(output_blob[object_index + 1]) * 2 - 0.5 + row)*stride;
				float w = pow(sigmoid_function(output_blob[object_index + 2]) * 2, 2)*anchors[anchor_index + c * 2];
				float h = pow(sigmoid_function(output_blob[object_index + 3]) * 2, 2)*anchors[anchor_index + c * 2 + 1];
				float max_prob = -1;
				int class_index = -1;
                                printf("side_hqqqq!!!: %f %f %f %f %f \n",conf, x, y, w, h);
				// 解析类别输出的网络宽度是类别数 5
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
				prob.push_back((float)conf);
				rect.push_back(Rect(x1, y1, x2 - x1, y2 - y1));

                                //printf("Confidence: %d %d %f %f %f %f\n", classIds.size(), classIds[0], x, y, w, h);
                                //printf("------DETECTION------\nLabel = %d\nConfidence = %.1f\nXmin = %.1f\nYmin = %.1f\nWidth = %.1f\nHeight = %.1f\n---------------------", "aa", confidences, x1, y1, x2 - x1, y2 - y1);
			}
		}
	}
*/



    //获得最终检测结果
    vector<int> final_id;
    dnn::NMSBoxes(origin_rect, origin_rect_cof, _cof_threshold, _nms_area_threshold, final_id);

    for (int i = 0; i < final_id.size(); ++i) {
        Rect resize_rect = origin_rect[final_id[i]];
        float angle = rotate_angle[final_id[i]];
        detected_objects.push_back(Object{ origin_rect_cof[final_id[i]], classIds[final_id[i]], resize_rect, angle});
    }
    return true;
}


double Rotate_Detector::sigmoid(double x)
{
    return (1 / (1 + exp(-x)));
}

vector<int> Rotate_Detector::get_anchors(int net_grid)
{
    vector<int> anchors(6);
    int         a80[6] = { 10, 13, 16, 30, 33, 23 };
    int         a40[6] = { 30, 61, 62, 45, 59, 119 };
    int         a20[6] = { 116, 90, 156, 198, 373, 326 };
    if (net_grid == 80) {
        anchors.insert(anchors.begin(), a80, a80 + 6);
    } else if (net_grid == 40) {
        anchors.insert(anchors.begin(), a40, a40 + 6);
    } else if (net_grid == 20) {
        anchors.insert(anchors.begin(), a20, a20 + 6);
    }
    return anchors;
}




int Rotate_Detector::get_anchor_index(int scale_w, int scale_h) {
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

float Rotate_Detector::get_stride(int scale_w, int scale_h) {
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

float Rotate_Detector::sigmoid_function(float a)
{
	float b = 1. / (1. + exp(-a));
	return b;
}


