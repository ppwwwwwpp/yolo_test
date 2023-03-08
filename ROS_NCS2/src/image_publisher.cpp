#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

int main(int argc, char *argv[])
{

    // provjera da li je dan video
    if (argc == 1)
    {
        std::cerr << "Not enough input arguments, specifiy input video path!" << std::endl;
        return 1;
    }
    
    // inicijalizacija node-a i potrebnih stvari za slanje slike
    ros::init(argc, argv, "image_publisher");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("video/image", 1);
        
    std::string video_name(argv[1]);
    cv::VideoCapture vc(video_name);
    // provjera da li se moze čitati video
    if (!vc.isOpened())
    {
        std::cerr << "Couldn't open the video!" << std::endl;
        return 1;
    }
    
    cv::Mat frame;
    sensor_msgs::ImagePtr msg;    
    ros::Rate loop_rate(10);
    
    while (ros::ok())
    {
        // čitanje frame-a
        if (!vc.read(frame))
        {
            // šaljemo prazan frame za kraj
            cv::Mat empty_frame;
            pub.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", empty_frame).toImageMsg());
            std::cout << "Video has finished!" << std::endl;
            vc.release();
            return 0;
        }
        // msg je tip sensor_msgs::ImagePtr, dakle pointer
        msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
        pub.publish(msg);
        loop_rate.sleep();
    }
    return 0;
}
