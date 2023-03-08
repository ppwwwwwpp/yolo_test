//
// Created by 武智辉 on 2017/6/6.
//
#include "img_similar.h"

#include <bitset>

int FP_SIZE = 64;

bool set_fp_size(int size){
    if(size < 5 || size > 300){
        return false;
    }
    FP_SIZE = size;
    return true;
}


int _get_threshold(Mat img){
    if (img.channels() != 1) {
        return -1;
    }
    int grey_n[256]={0};
    for (int i=0; i < img.rows; i++){
        for(int j=0; j < img.cols; j++){
            grey_n[img.at<uchar>(i, j)]++;
        }
    }
    int total = (int)img.total();
    float sum = 0;
    for (int t=0 ; t<256 ; t++) sum += t * grey_n[t];
    float sumb = 0;
    int wb = 0;
    int wf = 0;

    float var_max = 0;
    int threshold = 0;

    for (int t=0 ; t<256 ; t++) {
        wb += grey_n[t];               // Weight Background
        if (wb == 0) continue;

        wf = total - wb;                 // Weight Foreground
        if (wf == 0) break;

        sumb += (float) (t * grey_n[t]);

        float mb = sumb / wb;            // Mean Background
        float mf = (sum - sumb) / wf;    // Mean Foreground

        // Calculate Between Class Variance
        float var_between = (float)wb * (float)wf * (mb - mf) * (mb - mf);

        // Check if new maximum found
        if (var_between > var_max) {
            var_max = var_between;
            threshold = t;
        }
    }
    return threshold;
}


Mat get_fp(Mat img){
    Mat dst;
//    img = pbcvt::fromNDArrayToMat(obj);
    cvtColor(img, dst, CV_BGR2GRAY);
    int width = dst.size[1];  // notice 0 for height, 1 for width
    int height = dst.size[0];
    Mat sq_grey, std_grey, bw_img;
    if (width < height){
        sq_grey = dst(Rect(0, (height - width) / 2, width, width));
    }
    else {
        sq_grey = dst(Rect((width - height) / 2, 0, height, height));
    }
    resize(sq_grey, std_grey, Size(FP_SIZE, FP_SIZE));
    int th = _get_threshold(std_grey);
    threshold(std_grey, bw_img, th, 255, THRESH_BINARY);
//    PyObject* ret = pbcvt::fromMatToNDArray(bw_img);
//    return ret;
    return bw_img;
}


double calc_similarity(Mat fp1, Mat fp2){
    Mat ori_diff = ~(fp1 ^ fp2), blur_diff;
    blur(ori_diff, blur_diff, Size(FP_BLUR, FP_BLUR));
    double ori_diffn=0.,blur_diffn=0.;
    for (int i=0; i < ori_diff.rows; i++){
        for(int j=0; j < ori_diff.cols; j++){
            if(ori_diff.at<uchar>(i, j)<80) ori_diffn++;
            if(blur_diff.at<uchar>(i, j)<80) blur_diffn++;
        }
    }
    double similar = 1 - (ori_diffn * 0.2 + blur_diffn * 0.8)/ori_diff.total();
//    cout << "similarity: " << similar << endl;
    return similar;
}


char bit4_2_hexchar(std::string s){
    if(s.size() != 4){
        throw std::exception();
    }
    std::bitset<4> a(s);
    std::stringstream res;
    res << std::hex << std::uppercase << a.to_ulong();
    return res.str().c_str()[0];
}


static std::bitset<PART_LEN> get_bytes(Mat fp){
    std::bitset<PART_LEN> res;
    for(int i=0; i < fp.rows; i++){
        for(int j=0; j < fp.cols; j++){
            res[i * fp.cols + j] = fp.at<bool>(i, j);
        }
    }
    return res;
}

void get_fp_strs(Mat fp, std::string res[4]){
    std::string bits;
    Mat fps[4];
    std::bitset<PART_LEN> bitss[4];

    for(int i=0; i < 4; i++) {
        fps[i] = fp(Rect(i % 2 * PART_SIZE, i / 2 * PART_SIZE, PART_SIZE, PART_SIZE));
        bitss[i] = get_bytes(fps[i]);
    }
    for(int i=0; i < 4; i++){
        bits = "";
        for(int x=0; x < (4 - (PART_LEN * 3 % 4)) % 4; x++){
            bits += "0";
        }
        for(int j=0; j < 4; j++){
            if(j==i) continue;
            bits += bitss[j].to_string();
        }
        res[i].resize((u_long)PART_FP_LEN);
        for(uint k=0; k < PART_FP_LEN; k++)
            res[i][k] = bit4_2_hexchar(bits.substr(k * 4, 4));
    }
}
