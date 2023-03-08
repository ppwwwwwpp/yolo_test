#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <inference_engine.hpp>

#include "image_subscriber.hpp"
#include "util.hpp"
#include "ros_intel_stick/DetectionObject.h"

// konstanta koja predstavlju threshold vjerojatnost ispod koje
// se objekti neće razmatrati. Promijeniti po potrebi.
#define THRESHOLD 0.5
// konstanta koja predstavlja threshold "Intersection over union" poviše koje
// se objekti neće razmatrati, npr. ako se dva objekta preklapaju 50% onaj
// čija je preciznost manja uopće neće biti razmatran jer se pretpostavlja da 
// su to isti objekti (što ne mora nužno biti). Promijeniti po potrebi.
#define IOU 0.9

using namespace InferenceEngine;

int main(int argc, char *argv[])
{
    // provjera da li je zadan path do xml datoteke
    //if (argc == 1)
    //{
    //    std::cerr << "Not enough input arguments, specifiy xml filepath!" << std::endl;
    //    return 1;
    //}
    
    // iniciranje ros node-a
    ros::init(argc, argv, "image_subscriber");
    ros::NodeHandle nh;
    
    // iniciranje subscriber-a
    ImageSubscriber image_subscriber;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("video/image", 1, &ImageSubscriber::image_callback, &image_subscriber);
    // iniciranje publisher-a za slanje informacija o objektima
    ros::Publisher object_pub = nh.advertise<ros_intel_stick::DetectionObject>("object_detection", 1);
    
    ros::Rate loop_rate(30);

    // iniciranje inference engine-a
    InferencePlugin plugin = PluginDispatcher().getPluginByDevice("MYRIAD");    
    CNNNetReader net_reader;
    // ucitavamo xml datoteku modela
    //std::string model_name(argv[1]);
    std::string model_name = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/best.xml";
    std::string bin_file = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/best.bin";
    net_reader.ReadWeights(bin_file);
    net_reader.ReadNetwork(model_name);
    // postavljamo batch size na 1
    ROS_ERROR("222222222222");
    //net_reader.getNetwork().setBatchSize(1);
    CNNNetwork nework = net_reader.getNetwork();
    ROS_ERROR("44444444");  
    nework.setBatchSize(1);
    ROS_ERROR("333333");  
    // put do bin datoteke, pretpostavka da se xml i bin nalaze u istom direktoriju
    // i da imaju potpuno isto ime (osim tipa datoteke, bin umjesto xml)
    //std::string bin_file = model_name.erase(model_name.rfind('.')) + ".bin";
    //std::string bin_file = "/home/wl/test_ws/darknet/src/yolov7-ros/weights/best.bin";
    // učitavamo težine mreže iz bin datoteke
    //net_reader.ReadWeights(bin_file);

    // mapa s informacijamo o ulazu u mrežu, samo jedan ulaz u yolov3
    InputsDataMap input_info(net_reader.getNetwork().getInputsInfo());
    // uzimamo referencu na pointer na ulaz i ime ulaza, potrebno za kasnije
    InputInfo::Ptr &input = input_info.begin() -> second;
    std::string input_name = input_info.begin() -> first;
    // postavljamo preciznost ulaza na unsigned8, za piksele
    input -> setPrecision(Precision::U8);
    // postavljamo layout ulaza na batch_size-channels-height-width
    input -> getInputData() -> setLayout(Layout::NCHW);
    
    // mapa s informacijama o izlazima iz mreže, yolov3 ima 3 izlaza
    OutputsDataMap output_info(net_reader.getNetwork().getOutputsInfo());
    // iteriramo po izlazima te postavljamo izlaznu preciznost i layout.
    for (auto &output : output_info)
    {
        output.second -> setPrecision(Precision::FP32);
        output.second -> setLayout(Layout::NCHW);
    }
    
    // loading modela na stick
    ExecutableNetwork network = plugin.LoadNetwork(net_reader.getNetwork(), {});
    // pointeri na zahtjev za inferenciju, koristimo kad mreža obavlja prolaz
    InferRequest::Ptr async_infer_request_next = network.CreateInferRequestPtr();
    InferRequest::Ptr async_infer_request_curr = network.CreateInferRequestPtr();
    
    cv::Mat frame;
    // ovaj dio radimo kako bi osigurali da frame nije prazan,
    // ne zelimo poslati praznu matricu sticku
    while (ros::ok())
    {
        ros::spinOnce();
        frame = image_subscriber.get_frame();
        if (!frame.empty())
        {
            break;
        }
    }
    
    // dobivamo visinu i širinu, trebat će nam za obradu
    cv::Size size = frame.size();
    const size_t width  = (size_t) size.width;
    const size_t height = (size_t) size.height;
    
    bool first_frame = true;
    // glavna petlja, obavljamo asinkroni način rada, prvo pošaljemo jedan inference upit
    // dok se on obavlja pošaljemo i drugi upit, obrađuju se dva frame-a "istovremeno".
    while (ros::ok())
    {   
        // prvi frame, pošaljemo dva upita
        if (first_frame)
        {
            // ovaj upit se šalje samo za prvi frame
            FrameToBlob(frame, async_infer_request_curr, input_name);
            async_infer_request_curr -> StartAsync();
            // spin jednom da pozovemo callback i dobijemo novi frame.
            ros::spinOnce();
            frame = image_subscriber.get_frame();
            if (frame.empty())
            {
                break;
            }
            first_frame = false;
        }
        
        // ovaj upit se uvijek šalje
        FrameToBlob(frame, async_infer_request_next, input_name);
        async_infer_request_next -> StartAsync();
        
        // ispitujemo da li je upit gotov i čekamo dok nije gotov
        if (OK == async_infer_request_curr -> Wait(IInferRequest::WaitMode::RESULT_READY))
        {
            // procesiranje izlaza izvršenog zahtjeva
            //unsigned long resized_im_h = input_info.begin() -> second.get() -> getDims()[0];
            //unsigned long resized_im_w = input_info.begin() -> second.get() -> getDims()[1];
            unsigned long resized_im_h = 640;
            unsigned long resized_im_w = 640;
            std::vector<DetectionObject> objects;
            // parsiranje izlaza
            for (auto &output : output_info)
            {
                auto output_name = output.first;
                CNNLayerPtr layer = net_reader.getNetwork().getLayerByName(output_name.c_str());
                Blob::Ptr blob = async_infer_request_curr -> GetBlob(output_name);
                ParseYOLOV3Output(layer, blob, resized_im_h, resized_im_w, height, width, THRESHOLD, objects);
            }
            // filtriranje preklapajućih okvira
            std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
            for (size_t i = 0; i < objects.size(); ++i)
            {
                if (objects[i].confidence == 0)
                {
                    continue;
                }
                for (size_t j = i + 1; j < objects.size(); ++j)
                {
                    if (IntersectionOverUnion(objects[i], objects[j]) >= IOU)
                    {
                        objects[j].confidence = 0;
                    }
                }
            }
            
            //std::cout << "NOVI NIZ OBJEKATA" << std::endl;
            ros_intel_stick::DetectionObject msg;
            // slanje vrijednosti
            for (auto &object : objects)
            {
                if (object.confidence < THRESHOLD)
                {
                    continue;
                }
                /*
                std::cout << object.confidence << " "
                          << object.class_id << " "
                          << object.xmax << " "
                          << object.xmin << " "
                          << object.ymax << " "
                          << object.ymin << " " << std::endl;
                */
                
                msg.confidence = object.confidence;
                msg.class_id = object.class_id;
                msg.xmax = object.xmax;
                msg.xmin = object.xmin;
                msg.ymax = object.ymax;
                msg.ymin = object.ymin;
                object_pub.publish(msg);
            }
        }
        
        ros::spinOnce();
        //čitamo sljedeći frame, ako je prazan to znači da smo prestali
        // primati podatke iz videa/kamere.
        frame = image_subscriber.get_frame();
        if (frame.empty())
        {
            std::cout << "Last frame !" << std::endl;
            break;
        }
        
        // glavni dio za asinkroni način rada, mijenjamo pointere na upite
        // if izjava, koja provjerava da li je upit gotov, svaki prolaz kroz petlju
        // "propituje" drugi pointer.
        async_infer_request_curr.swap(async_infer_request_next);       
        loop_rate.sleep();
    }
    
    return 0;
}

