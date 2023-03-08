#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <inference_engine.hpp>

using namespace InferenceEngine;

#define YOLO_SCALE_13 13
#define YOLO_SCALE_26 26
#define YOLO_SCALE_52 52

/*
Ovdje se nalaze funkcije koje su preuzete od Intela iz njihovih primjera
služe za pripremu ulaznih vrijednosti, parsiranje izlaza, "Intersection over union" itd.
*/

template <typename T>
void matU8ToBlob(const cv::Mat &orig_image, Blob::Ptr &blob, int batch_index=0)
{
    SizeVector blob_size = blob -> getTensorDesc().getDims();
    const size_t width = blob_size[3];
    const size_t height = blob_size[2];
    const size_t channels = blob_size[1];
    T *blob_data = blob -> buffer().as<T*>();

    cv::Mat resized_image(orig_image);
    if (static_cast<int>(width) != orig_image.size().width ||
        static_cast<int>(height) != orig_image.size().height)
    {
        cv::resize(orig_image, resized_image, cv::Size(width, height));
    }

    int batch_offset = batch_index * width * height * channels;

    for (size_t c = 0; c < channels; c++)
    {
        for (size_t  h = 0; h < height; h++)
        {
            for (size_t w = 0; w < width; w++)
            {
                blob_data[batch_offset + c * width * height + h * width + w] = resized_image.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}


void FrameToBlob(const cv::Mat &frame, InferRequest::Ptr &infer_request, const std::string &input_name)
{
    /* Resize and copy data from the image to the input blob */
    Blob::Ptr frame_blob = infer_request -> GetBlob(input_name);
    matU8ToBlob<uint8_t>(frame, frame_blob);
}


struct DetectionObject
{
    int xmin;
    int ymin;
    int xmax;
    int ymax;
    int class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale)
    {
        this -> xmin = static_cast<int>((x - w / 2) * w_scale);
        this -> ymin = static_cast<int>((y - h / 2) * h_scale);
        this -> xmax = static_cast<int>(this->xmin + w * w_scale);
        this -> ymax = static_cast<int>(this->ymin + h * h_scale);
        this -> class_id = class_id;
        this -> confidence = confidence;
    }

    bool operator <(const DetectionObject &s2) const
    {
        return this -> confidence < s2.confidence;
    }
    
    bool operator >(const DetectionObject &s2) const
    {
        return this -> confidence > s2.confidence;
    }
};


static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry)
{
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}


void ParseYOLOV3Output(const CNNLayerPtr &layer,
                       const Blob::Ptr &blob,
                       const unsigned long resized_im_h,
                       const unsigned long resized_im_w,
                       const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold,
                       std::vector<DetectionObject> &objects)
{
                       
    // --------------------------- Validating output parameters -------------------------------------
    if (layer->type != "RegionYolo")
    {
        throw std::runtime_error("Invalid output type: " + layer->type + ". RegionYolo expected");
    }
    const int out_blob_h = static_cast<int>(blob -> getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob -> getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w)
    {
        throw std::runtime_error("Invalid size of output " + layer->name +
                                 " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
                                 ", current W = " + std::to_string(out_blob_h));
    }
        
    // --------------------------- Extracting layer parameters -------------------------------------
    auto num = layer -> GetParamAsInt("num");
    try 
    {
        num = layer -> GetParamAsInts("mask").size();
    }
    catch (...) {}
    
    auto coords = layer -> GetParamAsInt("coords");
    auto classes = layer -> GetParamAsInt("classes");
    std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0,
                                  45.0, 59.0, 119.0, 116.0, 90, 156.0, 198.0, 373.0, 326.0};
    try 
    {
        anchors = layer -> GetParamAsFloats("anchors");
    }
    catch (...) {}
    
    auto side = out_blob_h;
    int anchor_offset = 0;
    switch (side)
    {
        case YOLO_SCALE_13:
            anchor_offset = 2 * 6;
            break;
        case YOLO_SCALE_26:
            anchor_offset = 2 * 3;
            break;
        case YOLO_SCALE_52:
            anchor_offset = 2 * 0;
            break;
        default:
            throw std::runtime_error("Invalid output size");
    }
    auto side_square = side * side;
    const float *output_blob = blob -> buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
    
    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i)
    {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < num; ++n)
        {
            int obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords);
            int box_index = EntryIndex(side, coords, classes, n * side * side + i, 0);
            float scale = output_blob[obj_index];
            if (scale < threshold)
            {
                continue;
            }
            double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
            double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
            double height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n];
            for (int j = 0; j < classes; ++j)
            {
                int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
                float prob = scale * output_blob[class_index];
                if (prob < threshold)
                {
                    continue;
                }
                DetectionObject obj(x,
                                    y,
                                    height,
                                    width,
                                    j,
                                    prob,
                                    static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                                    static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
        }
    }
}


double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2)
{
    double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
    double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
    {
        area_of_overlap = 0;
    }
    else
    {
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;
    }
    double box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
    double box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
    double area_of_union = box_1_area + box_2_area - area_of_overlap;
    return area_of_overlap / area_of_union;
}

