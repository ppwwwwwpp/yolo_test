//
// Created by 武智辉 on 2017/6/6.
//

#ifndef IMG_SIMILAR_IMG_SIMILAR_H
#define IMG_SIMILAR_IMG_SIMILAR_H
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

extern int FP_SIZE;
const int FP_BLUR = 3;
const int PART_SIZE = 3;
const int PART_LEN = PART_SIZE * PART_SIZE;
const int PART_FP_LEN = (PART_LEN * 3 + 3) / 4;

bool set_fp_size(int size);

int _get_threshold(Mat img);

Mat get_fp(Mat img);

double calc_similarity(Mat fp1, Mat fp2);

void get_fp_strs(Mat fp, std::string res[4]);

#endif //IMG_SIMILAR_IMG_SIMILAR_H
