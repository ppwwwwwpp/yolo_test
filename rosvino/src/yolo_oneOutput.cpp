#include "yolo_oneOutput.h"

yoloOneOutput::yoloOneOutput(){
    //string xml_path = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/yolov5s.xml";
    //string bin_path = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/yolov5s.bin";
    string xml_path = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/best-tiny.xml";
    string bin_path = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/best-tiny.bin";
    init(xml_path,bin_path, 0.25, 0.5);

}

yoloOneOutput::~yoloOneOutput(){}
string names2[] = {"nomask","mask"};



bool yoloOneOutput::parse_yolov5(const Blob::Ptr &blob, int net_grid, float cof_thr, vector<Rect>& o_rect, vector<float>& o_rect_cof, vector<int>& classIds)
{
    vector<int> anchors = get_anchors(net_grid);
    LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
    const float *output_blob = blobMapped.as<float *>();

    int item_size = 7;
    for(int n=0; n<3; ++n)
    {
        for(int i=0; i<net_grid; ++i)
        {
            for(int j=0; j<net_grid; ++j)
            {
                double box_prob = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 4];
                box_prob = sigmoid(box_prob);
                if(box_prob < cof_thr)
                    continue;
                
                //注意此处输出为中心点坐标,需要转化为角点坐标
                double x = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 0];
                double y = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 1];
                double w = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 2];
                double h = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j *item_size+ 3];
               
                double max_prob = 0;
                int idx = 0;
                for(int t=5; t<7; ++t){
                    double tp = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + t];
                    tp = sigmoid(tp);
                    if(tp > max_prob){
                        max_prob = tp;
                        idx = t;
                    }
                }
                float cof = box_prob * max_prob;
                if(cof < cof_thr)
                    continue;

                x = (sigmoid(x) * 2 - 0.5 + j) * 640.0f / net_grid;
                y = (sigmoid(y) * 2 - 0.5 + i) * 640.0f / net_grid;
                int left = n * 2;
                int right = n * 2 + 1;
                w = pow(sigmoid(w) * 2, 2) * anchors[left];
                h = pow(sigmoid(h) * 2, 2) * anchors[right];
                double r_x = x - w / 2;
                double r_y = y - h / 2;
                Rect rect = Rect(round(r_x), round(r_y), round(w), round(h));
                o_rect.push_back(rect);
                o_rect_cof.push_back(cof);
                classIds.push_back(idx - 5);
            }
        }
    }
    return true;
}

bool yoloOneOutput::init(string xml_path, string bin_path, double cof_threshold, double nms_area_threshold)
{
    _xml_path = xml_path;
    _bin_path = bin_path;
    _cof_threshold = cof_threshold;
    _nms_area_threshold = nms_area_threshold;
    Core ie;
    //查询支持硬件设备
    std::vector<std::string> availableDev = ie.GetAvailableDevices();
		for (int i = 0; i < availableDev.size(); i++) {
			cout << "supported device name: " << availableDev[i].c_str() << endl;
		}

    //auto cnnNetwork = ie.ReadNetwork(_xml_path, _bin_path);
    //auto cnnNetwork = ie.ReadNetwork(_xml_path);
    InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(_xml_path);
    cnnNetwork.setBatchSize(1);
    // 输入设置
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    InputInfo::Ptr& input = inputInfo.begin()->second;
    _input_name = inputInfo.begin()->first;
    //cout << "input_name: " << _input_name << endl;
    input->setPrecision(Precision::FP32);
    input->getInputData()->setLayout(Layout::NCHW);
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    SizeVector& inSizeVector = inputShapes.begin()->second;
    cnnNetwork.reshape(inputShapes);
    // 输出设置
    //_outputinfo = OutputsDataMap(cnnNetwork.getOutputsInfo());
    _outputinfo = cnnNetwork.getOutputsInfo();
    for (auto &output : _outputinfo) 
    {
        //cout << "output_name: " << output.first << endl;
        output.second->setPrecision(Precision::FP32);
    }
    auto start = chrono::high_resolution_clock::now();
    _network =  ie.LoadNetwork(cnnNetwork, "CPU");
    auto end = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    cout<<"use_loadnetwork "<<diff.count()<<" s" << endl;

    return true;
}

bool yoloOneOutput::process_frame(Mat& inframe, vector<Object>& detected_objects)
{

    resize(inframe, inframe, Size(640, 640));
    cvtColor(inframe, inframe, COLOR_BGR2RGB);
    size_t img_size = 640 * 640;

    InferRequest infer_request = _network.CreateInferRequest();
    Blob::Ptr frameBlob = infer_request.GetBlob(_input_name);

    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();
    float* blob_data = blobMapped.as<float*>();

    for(size_t row = 0; row < 640; row++) {
        for(size_t col = 0; col < 640; col++) {
            for(size_t ch = 0; ch < 3; ch++) {
                blob_data[img_size * ch + row * 640 + col] = float(inframe.at<Vec3b>(row, col)[ch]) / 255.0f;
            }
        }
    }

    auto start = chrono::high_resolution_clock::now();
    infer_request.Infer();
    auto end = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    cout<<"use_infer_request "<<diff.count()<<" s" << endl;


    vector<Rect> origin_rect;
    vector<float> origin_rect_cof;
    vector<int> classIds;
    int s[3] = {80, 40, 20};
    int i = 0;

    for (auto &output : _outputinfo) {
        auto output_name = output.first;
        //cout << "output_name12: " << output_name << endl;
        Blob::Ptr blob = infer_request.GetBlob(output_name);
        parse_yolov5(blob, s[i], _cof_threshold, origin_rect, origin_rect_cof, classIds);
        ++i;
        if (i == 3)
        {

            break;
        }
    }

    vector<int> final_id;
    dnn::NMSBoxes(origin_rect, origin_rect_cof, _cof_threshold, _nms_area_threshold, final_id);

    for(int i=0; i<final_id.size(); ++i){
       // cout << "final_id: " << final_id[i] << endl;
        Rect resize_rect = origin_rect[final_id[i]];
        detected_objects.push_back(Object{classIds[final_id[i]],origin_rect_cof[final_id[i]], resize_rect});
    }

    return true;
}

double yoloOneOutput::sigmoid(double x){
    return (1 / (1 + exp(-x)));
}

vector<int> yoloOneOutput::get_anchors(int net_grid){
    vector<int> anchors(6);
//    int a80[6] = {10, 13, 16, 30, 33, 23};
//    int a40[6] = {30, 61, 62, 45, 59, 119};
//    int a20[6] = {116, 90, 156, 198, 373, 326}; 
    int a80[6] = {10, 13, 16, 30, 33, 23};
    int a40[6] = {30, 61, 62, 45, 59, 119};
    int a20[6] = {116, 90, 156, 198, 373, 326}; 
    if(net_grid == 80){
        anchors.insert(anchors.begin(), a80, a80 + 6);
    }
    else if(net_grid == 40){
        anchors.insert(anchors.begin(), a40, a40 + 6);
    }
    else if(net_grid == 20){
        anchors.insert(anchors.begin(), a20, a20 + 6);
    }
    return anchors;
}
