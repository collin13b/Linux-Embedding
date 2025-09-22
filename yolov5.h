#ifndef __YOLOV5_H
#define __YOLOV5_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>   
#include "/home/cat/linuxAi/rk3588prj/linuxlLearning/3rdparty/librknn_api/include/rknn_api.h"
#include "RgaUtils.h"
#include "im2d.h"
#include "rga.h"

using namespace std;
using namespace cv;

class yolov5s{ 

public:
    yolov5s(const char* model_path,int nup_index);
    ~yolov5s();

    //模型高宽，和通道数
    int model_width;
    int model_height;
    int model_channel;

    //输入的图像参数
    int img_width;
    int img_height;
    int img_channel;


    //模型推理函数
    int inference_image(const Mat &origin_img);
    int draw_result(const Mat &origin_img);//画框函数
private:
    rknn_context context;
    unsigned int model_size;
    rknn_tensor_attr input_attr;
    rknn_tensor_attr output_attr;
    rknn_input_output_num io_num;
    rknn_sdk_version version;
    vector<rknn_tensor_attr> input_attrs;
    vector<rknn_tensor_attr> output_attrs;
    
    unsigned char *model_data;
    unsigned char *load_model(const char *model_path,unsigned int &model_size);
};


#endif 