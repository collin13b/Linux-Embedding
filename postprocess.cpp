#include "postprocess.h"
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
float anchor0[6] = {10, 13, 16, 30, 33, 23};
float anchor1[6] = {30, 61, 62, 45, 59, 119};
float anchor2[6] = {116, 90, 156, 198, 373, 326};

struct Probaryy
{
    float conf;
    int index;
};
static vector<string> labels;
// Sigmoid 函数：计算输入值的 sigmoid 结果 激活函数 sigmod 将结果映射到【0,1】之间
static float sigmoid(float x)
{
    float y = 1 / (1 + exp(-1));
    return y;
}
//反激活函数 sigmod 将结果映射到【-♾️，+♾️】之间
static float unsigmoid(float y)
{
    float x = -1.0f * logf(1.0f / y - 1);
    return x;
}
//-----------------排序函数，对将执行度从高到低排序-------------------
static int sort_desecding(vector<Probaryy> &p_arr)
{
    sort(p_arr.begin(),p_arr.end(),[](const Probaryy &a,const Probaryy &b){return a.conf > b.conf;});
    return 0;
}
//-----------------iou计算交并比-------------------//
static float caculateIou(float x1min,float y1min,float x1max,float y1max,
                        float x2min,float y2min,float x2max,float y2max)
{
    float w = fmax(0.f,fmin(x1max,x2max) - fmax(x1min,x2min) + 1.0) ;
    float h = fmax(0.f,fmin(y1max,y2max) - fmax(y1min,y2min) + 1.0) ;
    float i = w * h;
    float u = (x1max - x1min + 1.0) * (y1max - y1min + 1.0) + (x2max - x2min + 1.0) * (y2max - y2min + 1.0) - i;
    float iou = u <= 0.f ? 0.f : (i / u);
    return iou;

}
//-----------------非极大值抑制-------------------///
/*Validcount: 有效检测框数量
boxs: 检测框坐标信息
class_id: 类别标识
indexArray: 输出的保留框索引
currtclass: 当前处理的类别
nms_threshold: NMS抑制阈值
遍历当前类别的所有检测框
按置信度排序
计算框间IoU重叠度
抑制重叠度超过阈值的框
返回保留的框索引
*/

static int nms(int Validcount,vector<float> &boxs,vector<int> &class_id,vector<int> &indexArray, int currtclass,float nms_threshold)
{
    for(int i = 0; i < Validcount; i++)
    {
        if(class_id[i] != currtclass || indexArray[i] == -1)
        {
            continue;
        }
        int n = indexArray[i];
        for(int j = i + 1; j < Validcount; j++)
        {
            int m = indexArray[j];
            if( m == -1 || class_id[j] != currtclass)
            {
                continue;
            }
            float xmin0 = boxs[n * 4];
            float ymin0 = boxs[n * 4 + 1];
            float xmax0 = boxs[n * 4 + 2] + xmin0;
            float ymax0 = boxs[n * 4 + 3] + ymin0;

            float xmin1 = boxs[m * 4];
            float ymin1 = boxs[m * 4 + 1];
            float xmax1 = boxs[m * 4 + 2] + xmin1;
            float ymax1 = boxs[m * 4 + 3] + ymin1;
            float iou = caculateIou(xmin0,ymin0,xmax0,ymax0,xmin1,ymin1,xmax1,ymax1);
            if(iou > nms_threshold)
            {
                indexArray[j] = -1;
            }
        }
    }
    return 0;
}
//---------------载入标签文件----------------
int read_lines(const char *label_path,vector<string> &labels,int max_lins)
{
    ifstream file(label_path);
    if(!file.is_open())
    {
        cout<<"open file error"<<endl;
        return -1;
    }
    string line;
    //循环一直读取 file 里面的内容
    while(getline(file,line))
    {
        labels.emplace_back(line);
        if(labels.size() >= max_lins)
        {
            break;
        }
    }
    return labels.size();
}
int LoadLbaelName(const char *filepath,vector<string> &labels,int labels_num)
{
    int line_nume = read_lines(filepath,labels,labels_num);

    cout<<"labels num:"<<line_nume<<endl;
    return line_nume;
}
static float deqnt_int8_to_f32(int8_t in_num,int32_t zp,float scale)
{
    float f = (float)(in_num - zp) * scale;
    return f;
}
static int8_t eqnt_f32_to_int8(float float_num,int32_t zp,float scale)
{
    int8_t out_num = (int8_t)(round(float_num / scale) + zp);
    return out_num;
}
/*
参数：
1. input：要处理的 buffer
2. anchor：锚框的长宽参数地址
3. grid_h、grid_w：单元网格数
4. model_height、model_width：模型要求的输入尺寸
5. stride：单元格的步长
6. boxes：存放检测框坐标
7. objProbs：存放目标置信度
8. classID：存放类别索引
9. box_threshold：过滤阈值
10. zp、scale：零点和缩放比例

处理模型输出的特征图数据(input)
结合锚框信息(anchor)进行解码
在指定网格尺寸(grid_h, grid_w)下
根据模型输入尺寸(model_height, model_width)和步长(stride)计算实际坐标
进行量化反量化处理(zp, scale)
筛选置信度高于阈值(box_threshold)的检测框
将结果分别存储到边界框坐标(boxes)、物体概率(objProbs)和类别ID(classID)向量中
*/
 int process(int8_t *input, float *anchor, int grid_h, int grid_w, int model_height, int model_width, int stride,
            vector<float> &boxes, vector<float> &objProbs, vector<int> &classID, float box_threshold, int32_t zp, float scale)
{
    int Validcount = 0;
    int grid_len = grid_h * grid_w;
    float box_using = unsigmoid(box_threshold);//将置信度变为原始值 在转化为 int8 类型
    int8_t box_int8 = eqnt_f32_to_int8(box_using,zp,scale);
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < grid_h; ++j)
        {
            for(int k = 0; k < grid_w; ++k)
            {
                //最终获取的是第i个网格、第j行、第k列位置处锚框的置信度值，并存储为int8_t类型
                int8_t box_anchor_conf = input[(i * BOX_NUM_SIZE + 4) * grid_len * i + grid_h * j + k];
                if(box_anchor_conf > box_int8)
                {
                    Validcount++;
                    //计算偏移值
                    int box_ofst = (i * BOX_NUM_SIZE + 4) * grid_len * i + grid_h * j + k;
                    int8_t *box_ptr = input + box_ofst;

                    //反量化和激活函数获取框和坐标信息
                    float box_x = sigmoid(deqnt_int8_to_f32(*box_ptr,zp,scale)) * 2 - 0.5;//将(0,1)范围映射到(-0.5, 1.5)
                    float box_y = sigmoid(deqnt_int8_to_f32(*(box_ptr + 1 * grid_len),zp,scale)) * 2 - 0.5;
                    float box_w = sigmoid(deqnt_int8_to_f32(*(box_ptr + 2 * grid_len),zp,scale)) * 2.0;//宽高缩放因子，通过*2.0放大范围
                    float box_h = sigmoid(deqnt_int8_to_f32(*(box_ptr + 3 * grid_len),zp,scale)) * 2.0;

                    //计算框的坐标
                    box_x = (box_x + k) * (float) stride;
                    box_y = (box_y + j) * (float) stride;
                    box_w = box_w * box_w *(float)anchor[i * 2];
                    box_h = box_h * box_h *(float)anchor[i * 2 + 1];

                    box_x = box_x - (box_w / 2.0);
                    box_y = box_y - (box_h / 2.0);

                    boxes.emplace_back(box_x);
                    boxes.emplace_back(box_y);
                    boxes.emplace_back(box_w);
                    boxes.emplace_back(box_h);
                    // 获取最大类别概率及对应的类别 ID
                    int8_t maxClassProb = *(box_ptr + 5 * grid_len);
                    int maxClassID = 0;
                    for(int a = 1; a < OBJ_CLASS_NUM; a++)
                    {
                        int8_t prob = *(box_ptr +(5 + a) * grid_len);
                        if(prob > maxClassProb)
                        {
                            maxClassProb = prob;
                            maxClassID = a;
                        }
                    }

                    objProbs.emplace_back(sigmoid(deqnt_int8_to_f32(maxClassProb,zp,scale)));
                    classID.emplace_back(maxClassID);
                }
            }
        }
    }
    return Validcount;
}

/*输入处理：接收三个int8量化的输出张量(output0/output1/output2)
参数配置：获取模型尺寸、NMS阈值、置信度阈值等参数
反量化：使用量化参数(qnt_zps和qnt_scales)将int8数据转换为float
坐标转换：通过scale_w和scale_h将检测框坐标映射到原始图像尺寸
结果筛选：应用置信度和NMS阈值过滤检测结果
输出结果：将处理后的检测结果存储在detect_result_group_ptr中
*/

inline static int clamp(float val,int min,int max){return val < min ? min : (val > max ? max : val);}


int postprocess(int8_t *output0,int8_t *output1,int8_t *output2,int model_height,int model_width,float nms_threshold,
                float box_threshold,float scale_w,float scale_h,vector<int32_t> &qnt_zps, vector<float> &qnt_scales,detect_result_group &result_group)
{
    //加载标签
    static bool label_flag = false;
    if(!label_flag)
    {
        int label_num = LoadLbaelName(LABEL_NAME_FILE,labels,OBJ_CLASS_NUM);
        label_flag = true;
    }

    vector<float> detect_boxes;
    vector<float> objProbss;
    vector<int> classIDs;
    //处理第一个输出
    int stride0 = 8;
    int grid_w0 = model_width / stride0;
    int grid_h0 = model_height / stride0;
    int Validcount0 = process(output0,anchor0,grid_h0,grid_w0,model_height,model_width,stride0,detect_boxes,objProbss,classIDs,box_threshold,qnt_zps[0],qnt_scales[0]);

    //处理第二个输出
    int stride1 = 16;
    int grid_w1 = model_width / stride1;
    int grid_h1 = model_height / stride1;
    int Validcount1 = process(output1,anchor1,grid_h1,grid_w1,model_height,model_width,stride0,detect_boxes,objProbss,classIDs,box_threshold,qnt_zps[1],qnt_scales[1]);

    //处理第三个输出
    int stride2 = 32;
    int grid_w2 = model_width / stride2;
    int grid_h2 = model_height / stride2;
    int Validcount2 = process(output2,anchor2,grid_h2,grid_w2,model_height,model_width,stride0,detect_boxes,objProbss,classIDs,box_threshold,qnt_zps[2],qnt_scales[2]);
    
    vector<int> indexarray;
    int validcount = Validcount0 + Validcount1 + Validcount2;
    if(validcount < 0)
    {
        return 0;
    }
    //把所有置信度和对应索引放入容器
    vector<Probaryy> prob_arr;
    for(int i = 0; i < validcount; i++)
    {
        Probaryy prob;
        prob.conf = objProbss[i];
        prob.index = i;
        prob_arr.emplace_back(prob);
    }
    //排序
    sort_desecding(prob_arr);
    indexarray.clear();
    objProbss.clear();

    //将排序好的目标置信度和对应索引放入容器
    for(int i = 0; i < validcount; i++)
    {
        objProbss.emplace_back(prob_arr[i].conf);
        indexarray.emplace_back(prob_arr[i].index);
    }

    //对每个类别进行非极大值抑制
    set<int> class_set(begin(classIDs),end(classIDs));
    for(const int &id : class_set)
    {
        nms(validcount,detect_boxes,classIDs,classIDs,id,nms_threshold);
    }
    //最后将结果放入结果结构体
    int count = 0;
    result_group.box_num = 0;
    for(int i = 0; i < validcount; i++)
    { 
        if(indexarray[i] == -1 || count >= MAX_OBJ_BOXS)
        {
            continue;
        }
        int n = indexarray[i];

        float xmin = detect_boxes[n * 4 + 0];
        float ymin = detect_boxes[n * 4 + 1];
        float xmax = detect_boxes[n * 4 + 2] + xmin;
        float ymax = detect_boxes[n * 4 + 3] + ymin;
        float box_conf = objProbss[i];
        int class_id = classIDs[n];

        //将坐标映射到原始图像尺寸
        result_group.result[count].box.xmin = (int)(clamp(xmin , 0, model_width) / scale_w);
        result_group.result[count].box.ymin = (int)(clamp(ymin , 0, model_height) / scale_h);
        result_group.result[count].box.xmax = (int)(clamp(xmax , 0, model_width) / scale_w);
        result_group.result[count].box.ymax = (int)(clamp(ymax , 0, model_height) / scale_h);
        result_group.result[count].box_conf = box_conf; 

        const char *label_temp = labels[class_id].c_str();
        strncpy(result_group.result[count].label,label_temp,32);

        count++;
        result_group.box_num = count;

    }
    return 0;
}