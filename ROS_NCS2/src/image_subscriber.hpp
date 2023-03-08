#pragma once

#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

class ImageSubscriber
{
    private:
    cv::Mat frame;
    
    public:
    void image_callback(const sensor_msgs::ImageConstPtr &);
    cv::Mat get_frame();
};

void ImageSubscriber::image_callback(const sensor_msgs::ImageConstPtr &msg)
{
    // pretvaramo primljenu poruku u cv::Mat
    frame = cv_bridge::toCvCopy(msg, "bgr8") -> image;
}

cv::Mat ImageSubscriber::get_frame()
{
    return frame;
}
