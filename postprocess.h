#ifndef __POSTPROCESS_H
#define __POSTPROCESS_H
//后处理函数声明
/* 后处理函数
nms 极大值抑制，置信框
 */
#include <stdint.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>
#include <map>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define ONJ_NUM_MAX_SIZE 64
#define OBJ_CLASS_NUM 80
#define LABEL_NAME_FILE "/home/cat/linuxAi/rk3588prj/linuxlLearning/model/coco_80_labels_list.txt"
#define BOX_NUM_SIZE (5+OBJ_CLASS_NUM)
#define MAX_OBJ_BOXS 60

#define BOX_THRESHOLD 0.5
#define NMS_THRESHOLD 0.5
using namespace std;

struct box_p
{
    int xmin;
    int ymin;
    int xmax;
    int ymax;
};
struct detect_result
{
    char label[32];
    float box_conf;
    box_p box;
};
struct detect_result_group
{
    int box_num;
    detect_result result[MAX_OBJ_BOXS];
};
int postprocess(int8_t *output0,int8_t *output1,int8_t *output2,int model_height,int model_width,float nms_threshold,float box_threshold,float scale_w,float scale_h,vector<int32_t> &qnt_zps, vector<float> &qnt_scales,detect_result_group &detect_result_group_ptr);

#endif