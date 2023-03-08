
#include <string>
#include <vector>
#include <iostream>

#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>

#include <opencv2/core/core.hpp>

#include <ngraph/ngraph.hpp>
#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>
#include "object_detection_yolo/monitors/presenter.h"
#include "object_detection_yolo/ocv_common.hpp"
#include "object_detection_yolo/common.hpp"
#include <opencv2/imgproc/imgproc.hpp>


#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>


#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>

//Eigen
#include <Eigen/Eigen>
#include <Eigen/StdVector>

using namespace InferenceEngine;

cv::Mat Color_pic;
cv::Mat Depth_pic;
bool init_flag = false;

double fx = 379.177978515625;
double fy = 378.29473876953125;
double cx = 314.47491455078125;
double cy = 246.75747680664062;

void frameToBlob(const cv::Mat& frame,InferRequest::Ptr& inferRequest,const std::string& inputName)
{
        Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
        matU8ToBlob<uint8_t>(frame, frameBlob);
}

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

struct DetectionObject {
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) {
        this->xmin = static_cast<int>((x - w / 2) * w_scale);
        this->ymin = static_cast<int>((y - h / 2) * h_scale);
        this->xmax = static_cast<int>(this->xmin + w * w_scale);
        this->ymax = static_cast<int>(this->ymin + h * h_scale);
        this->class_id = class_id;
        this->confidence = confidence;
    }

    bool operator <(const DetectionObject &s2) const {
        return this->confidence < s2.confidence;
    }
    bool operator >(const DetectionObject &s2) const {
        return this->confidence > s2.confidence;
    }
};

double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2) {
    double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
    double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
        area_of_overlap = 0;
    else
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;
    double box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
    double box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
    double area_of_union = box_1_area + box_2_area - area_of_overlap;
    return area_of_overlap / area_of_union;
}

class YoloParams {
    template <typename T>
    void computeAnchors(const std::vector<T> & mask) {
        std::vector<float> maskedAnchors(num * 2);
        for (int i = 0; i < num; ++i) {
            maskedAnchors[i * 2] = anchors[mask[i] * 2];
            maskedAnchors[i * 2 + 1] = anchors[mask[i] * 2 + 1];
        }
        anchors = maskedAnchors;
    }

public:
    int num = 0, classes = 0, coords = 0;
    std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0,
                                  156.0, 198.0, 373.0, 326.0};

    YoloParams() {}

    YoloParams(const std::shared_ptr<ngraph::op::RegionYolo> regionYolo) {
        coords = regionYolo->get_num_coords();
        classes = regionYolo->get_num_classes();
        anchors = regionYolo->get_anchors();
        auto mask = regionYolo->get_mask();
        num = mask.size();

        computeAnchors(mask);
    }
};

void ParseYOLOV3Output(const YoloParams &params, const std::string & output_name,
                       const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<DetectionObject> &objects) {

    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w)
        throw std::runtime_error("Invalid size of output " + output_name +
        " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
        ", current W = " + std::to_string(out_blob_h));

    auto side = out_blob_h;
    auto side_square = side * side;
    LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
    const float *output_blob = blobMapped.as<float *>();
    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < params.num; ++n) {
            int obj_index = EntryIndex(side, params.coords, params.classes, n * side * side + i, params.coords);
            int box_index = EntryIndex(side, params.coords, params.classes, n * side * side + i, 0);
            float scale = output_blob[obj_index];
            if (scale < threshold)
                continue;
            double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
            double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
            double height = std::exp(output_blob[box_index + 3 * side_square]) * params.anchors[2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * params.anchors[2 * n];
            for (int j = 0; j < params.classes; ++j) {
                int class_index = EntryIndex(side, params.coords, params.classes, n * side_square + i, params.coords + 1 + j);
                float prob = scale * output_blob[class_index];
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
        }
    }
}


void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
   cv_bridge::CvImagePtr cam_img;


   try {
       cam_img = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
   }catch(cv_bridge::Exception& e){
       ROS_ERROR("cv_bridge exception: %s",e.what());
       return;
   }
   if(!cam_img->image.empty())
   {
       Color_pic = cam_img->image.clone();
       if(Color_pic.cols!=640||Color_pic.rows!=480)
           cv::resize(Color_pic,Color_pic,cv::Size(640,480));
       //std::cout<<"received pic!"<<std::endl;

   }

}
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped> SyncPolicyImagePose;
typedef std::shared_ptr<message_filters::Synchronizer<SyncPolicyImagePose>> SynchronizerImagePose;
Eigen::Vector3d camera_pos;
Eigen::Quaterniond camera_q;
void depthPoseCallback(const sensor_msgs::ImageConstPtr& img,const geometry_msgs::PoseStampedConstPtr& pose)
{
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(img, img->encoding);
    if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, 1000);
      }
    cv_ptr->image.copyTo(Depth_pic);

    camera_pos(0) = pose->pose.position.x;
    camera_pos(1) = pose->pose.position.y;
    camera_pos(2) = pose->pose.position.z;
    camera_q = Eigen::Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x,
                                         pose->pose.orientation.y, pose->pose.orientation.z);
    //std::cout<<"recieved depth!"<<std::endl;

}

bool cap_sign = false;
int cap_count = 0;
void capCallback(const std_msgs::BoolConstPtr& msg)
{
    cap_sign = msg->data;
    cap_count++;
}

int main(int argc, char** argv){
    ros::init(argc, argv, "object_detection_yolo");
    ros::NodeHandle nh("~");
    ros::Publisher object_pub = nh.advertise<geometry_msgs::PoseStamped>("/object/pose",1);

    ros::Subscriber cap_sub = nh.subscribe("/pic_cap",1,capCallback);

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber img_sub = it.subscribe("/usb_cam/image_raw_throttled",1,imageCallback);

    std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> depth_sub_;
    std::shared_ptr<message_filters::Subscriber<geometry_msgs::PoseStamped>> pose_sub_;
    SynchronizerImagePose sync_image_pose_;

    depth_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh, "/camera/aligned_depth_to_color/image_raw", 50));
    pose_sub_.reset(new message_filters::Subscriber<geometry_msgs::PoseStamped>(nh, "/camera/pose", 25));
    sync_image_pose_.reset(new message_filters::Synchronizer<SyncPolicyImagePose>(SyncPolicyImagePose(100), *depth_sub_, *pose_sub_));
    sync_image_pose_->registerCallback(boost::bind(&depthPoseCallback, _1, _2));

    image_transport::Publisher img_pub = it.advertise("object_detection",1);
    std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

    std::vector<std::string> labels;
    labels.reserve(8);

    labels.push_back("danyaoxiang");
    labels.push_back("duijiangji");
    labels.push_back("micaibeibao");
    labels.push_back("micaishuihu");
    labels.push_back("miehuoqi");
    labels.push_back("qiangxie");
    labels.push_back("rentimoxing");
    labels.push_back("shoulei");


    cv::Mat frame ;//= Color_pic.clone();
    cv::Mat depth_frame;
    Eigen::Vector3d frame_pos;
    Eigen::Quaterniond frame_q;
    cv::Mat next_frame;

    int WID = 640;
    int HEI = 480;
    const size_t width  = (size_t) WID;
    const size_t height = (size_t) HEI;


    Core ie;
    CNNNetwork cnnNetwork;
    std::cout << ie.GetVersions("CPU") << std::endl;

    cnnNetwork = ie.ReadNetwork("/home/wl/test_ws/darknet/src/yolov7-ros/weights/best-sim.xml");
    cnnNetwork.setBatchSize(1);

    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());

    InputInfo::Ptr& input = inputInfo.begin()->second;
    auto inputName = inputInfo.begin()->first;
    input->setPrecision(Precision::U8);

    input->getInputData()->setLayout(Layout::NCHW);

    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    SizeVector& inSizeVector = inputShapes.begin()->second;
    inSizeVector[0] = 1;  // set batch to 1
    cnnNetwork.reshape(inputShapes);

    OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
    for (auto &output : outputInfo) {
        output.second->setPrecision(Precision::FP32);
//        output.second->setLayout(Layout::NCHW);
    }

    std::map<std::string, YoloParams> yoloParams;

/*    if (auto ngraphFunction = cnnNetwork.getFunction()) {
        for (const auto op : ngraphFunction->get_ops()) { 
            ROS_ERROR("66666666666666");
            auto outputLayer = outputInfo.find(op->get_friendly_name());
            if (outputLayer != outputInfo.end()) {
                auto regionYolo = std::dynamic_pointer_cast<ngraph::op::RegionYolo>(op);
                if (!regionYolo) {
                    throw std::runtime_error("Invalid output type: " +
                        std::string(regionYolo->get_type_info().name) + ". RegionYolo expected");
                }
                yoloParams[outputLayer->first] = YoloParams(regionYolo);

            }
        }
    }

    else {
        throw std::runtime_error("Can't get ngraph::Function. Make sure the provided model is in IR version 10 or greater.");
    }
*/
    std::cout<<"begin loading"<<std::endl;

    ExecutableNetwork network = ie.LoadNetwork(cnnNetwork, "CPU");
    std::cout<<"loading finished"<<std::endl;

    InferRequest::Ptr async_infer_request_curr = network.CreateInferRequestPtr();

    bool isAsyncMode = false;

    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
    auto total_t0 = std::chrono::high_resolution_clock::now();
    auto wallclock = std::chrono::high_resolution_clock::now();
    double ocv_render_time = 0;

    cv::Size graphSize{(int)640 / 4, 60};
    Presenter presenter("", (int)480 - graphSize.height - 10, graphSize);



    ros::Rate loop_rate(30);
    //cv::namedWindow("Object_Detection");

    while (ros::ok()) {
        ros::spinOnce();
        loop_rate.sleep();
        if(Color_pic.empty()) {
            std::cout<<"pic is empty!"<<std::endl;
            continue;
        }
        if(!init_flag)
            {
               init_flag =true;
            }
        else
            {
                 frame = Color_pic.clone();
                 depth_frame = Depth_pic.clone();
                 frame_pos = camera_pos;
                 frame_q = camera_q;
                 std::vector<std::string> class_name;
                 std::vector<Eigen::Vector3d> pos_list;
                 auto t0 = std::chrono::high_resolution_clock::now();

                 frameToBlob(frame, async_infer_request_curr,inputName);

                 auto t1 = std::chrono::high_resolution_clock::now();
                 double ocv_decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();

                 t0 = std::chrono::high_resolution_clock::now();

                 async_infer_request_curr->StartAsync();
                 if (OK == async_infer_request_curr->Wait(IInferRequest::WaitMode::RESULT_READY)) {
                     t1 = std::chrono::high_resolution_clock::now();
                     ms detection = std::chrono::duration_cast<ms>(t1 - t0);

                     t0 = std::chrono::high_resolution_clock::now();
                     ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
                     wallclock = t0;

                     t0 = std::chrono::high_resolution_clock::now();

                     presenter.drawGraphs(frame);
                     std::ostringstream out;
                     out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
                         << (ocv_decode_time + ocv_render_time) << " ms";
                     cv::putText(frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 255, 0));
                     out.str("");
                     out << "Wallclock time " << (isAsyncMode ? "(TRUE ASYNC):      " : "(SYNC, press Tab): ");
                     out << std::fixed << std::setprecision(2) << wall.count() << " ms (" << 1000.f / wall.count() << " fps)";
                     cv::putText(frame, out.str(), cv::Point2f(0, 50), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255));

                     out.str("");
                     out << "Detection time  : " << std::fixed << std::setprecision(2) << detection.count()
                         << " ms ("
                         << 1000.f / detection.count() << " fps)";
                     cv::putText(frame, out.str(), cv::Point2f(0, 75), cv::FONT_HERSHEY_TRIPLEX, 0.6,
                                 cv::Scalar(255, 0, 0));

                     const TensorDesc& inputDesc = inputInfo.begin()->second.get()->getTensorDesc();
                     unsigned long resized_im_h = getTensorHeight(inputDesc);
                     unsigned long resized_im_w = getTensorWidth(inputDesc);
                     std::vector<DetectionObject> objects;
                     for (auto &output : outputInfo) {
                         auto output_name = output.first;
                         Blob::Ptr blob = async_infer_request_curr->GetBlob(output_name);
                         ParseYOLOV3Output(yoloParams[output_name], output_name, blob, resized_im_h, resized_im_w, height, width, 0.5, objects);
                     }
                     std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
                     for (size_t i = 0; i < objects.size(); ++i) {
                         if (objects[i].confidence == 0)
                             continue;
                         for (size_t j = i + 1; j < objects.size(); ++j)
                             if (IntersectionOverUnion(objects[i], objects[j]) >= 0.4)
                                 objects[j].confidence = 0;
                     }

                     ROS_ERROR("wwwwwwwwww");
                     for (auto &object : objects) {
                         ROS_ERROR("qqqqqqqqqqqq: %f", object.confidence);
                         if (object.confidence < 0.5)
                             continue;
                         auto label = object.class_id;
                         float confidence = object.confidence;
                         if (confidence > 0.5) {
                             /** Drawing only objects when >confidence_threshold probability **/
                             std::ostringstream conf;
                             conf << ":" << std::fixed << std::setprecision(3) << confidence;
                             cv::putText(frame,
                                         (!labels.empty() ? labels[label] : std::string("label #") + std::to_string(label)) + conf.str(),
                                         cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin - 5)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                                         cv::Scalar(0, 0, 255));
                             cv::rectangle(frame, cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin)),
                                           cv::Point2f(static_cast<float>(object.xmax), static_cast<float>(object.ymax)), cv::Scalar(0, 0, 255));

                             if(depth_frame.empty()) continue;
                             auto m_t0 = std::chrono::high_resolution_clock::now();
                             int x1,y1,x2,y2;
                             x1 = static_cast<int>(object.xmin);
                             x1 = x1 >= 0 ? x1 : 0;

                             y1 = static_cast<int>(object.ymin);
                             y1 = y1 >= 0 ? y1 : 0;

                             x2 = static_cast<int>(object.xmax);
                             x2 = x2 <= 640 ? x2 : 640;

                             y2 = static_cast<int>(object.ymax);
                             y2 = y2 <= 480 ? y2 : 480;

                             int u,v;
                             u = (x1 + x2)/2;
                             v = (y1 + y2)/2;


                             double depth;
                             depth = double(depth_frame.at<uint16_t>(v,u))/1000;
                             //std::cout<<"depth: "<<depth<<std::endl;

                             Eigen::Matrix3d camera_r = frame_q.toRotationMatrix();
                             Eigen::Vector3d pt_cur, pt_world;

                             pt_cur(0) = (u - cx) * depth / fx;
                             pt_cur(1) = (v - cy) * depth / fy;
                             pt_cur(2) = depth;

                             pt_world = camera_r * pt_cur + frame_pos;
                             geometry_msgs::PoseStamped object_pos;
                             object_pos.header.stamp = ros::Time::now();
                             object_pos.header.frame_id = "map";
                             object_pos.pose.position.x = pt_world(0);
                             object_pos.pose.position.y = pt_world(1);
                             object_pos.pose.position.z = pt_world(2);

                             object_pub.publish(object_pos);

                             class_name.push_back(labels[label]);
                             pos_list.push_back(pt_world);



                             /*

                             cv::Mat depth_roi = depth_frame(cv::Rect(cv::Point2i(x1, y1),cv::Point2i(x2, y2)));
                             cv::Mat tmp_mean,tmp_sd;

                             double m,sd;
                             cv::meanStdDev(depth_roi,tmp_mean,tmp_sd);
                             m = tmp_mean.at<double>(0,0);
                             sd = tmp_sd.at<double>(0,0);
                             auto m_t1 = std::chrono::high_resolution_clock::now();
                             ms m_wall = std::chrono::duration_cast<ms>(m_t1 - m_t0);
                             std::cout<<"cost time: "<<std::setprecision(2) <<m_wall.count()<<" ms"<<std::endl;
                             std::cout<<"mean: "<<m/1000<<std::endl;
                             std::cout<<"sd: "<<sd/1000<<std::endl;
                             */
                         }
                     }
                     if(cap_sign)
                     {
                        if(class_name.empty()||pos_list.empty()) continue;
                        cap_sign = false;
                        cv::imwrite("/home/wl/test_ws/"+std::to_string(cap_count)+".jpg",frame);
                        std::string filename = "/home/wl/test_ws/" + std::to_string(cap_count)+".txt";
                        std::ofstream file;
                        file.open(filename.c_str());
                        file << std::fixed;
                        for(int i=0; i<class_name.size();i++)
                        {
                           file<<std::to_string(i)<<": "<<std::endl
                           <<"pos: "<<std::endl
                           <<" { x: "<<std::setprecision(6)<<pos_list[i](0)<<std::endl
                           <<"   y: "<<std::setprecision(6)<<pos_list[i](1)<<std::endl
                           <<"   z: "<<std::setprecision(6)<<pos_list[i](2)<<" }"<<std::endl
                           <<"timestamp: "<<std::setprecision(6)<<ros::Time::now().toSec()<<std::endl
                           <<"class name: "<<class_name[i]<<std::endl<<std::endl;
                        }
                         file.close();

                      }

                 }
                 cv::imshow("Detection results", frame);
                 cv::waitKey(3);

            }

    }


    return 0;

}
