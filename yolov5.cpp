#include "yolov5.h"
yolov5s::yolov5s(const char* model_path,int nup_index)
{
    int ret ;
    //载入模型
    this->model_size = 0;
    model_data = load_model(model_path,this->model_size);
    //将模型初始化加载到 rknn 中
    ret = rknn_init(&this->context, model_data, this->model_size , RKNN_FLAG_PRIOR_HIGH, NULL);
    if(ret  < 0)
    {
        printf("rknn_init error ret=%d\n",ret);
    }
    else
    {
        printf("rknn_init success!\n");
    }

    /* 对不同线程分配NPU，加速计算 */
    rknn_core_mask core_mask;
    if(nup_index == 0)
    {
        core_mask = RKNN_NPU_CORE_0;
    }
    else if(nup_index == 1)
    {
        core_mask = RKNN_NPU_CORE_1;
    } 
    else if(nup_index == 2)
    {
        core_mask = RKNN_NPU_CORE_2;
    }
    //设置NPU核心
    rknn_set_core_mask(this->context, core_mask);
    /* 够查询获取到模型输入输出信息、逐层运行时间、模型推理的总时间、
        SDK版本、内存占用信息、用户自定义字符串等信息 */
    ret = rknn_query(this->context,RKNN_QUERY_SDK_VERSION,&this->version,sizeof(this->version));
    if(ret < 0)
    {
        printf("rknn_query error ret=%d\n",ret);
    }
    ret = rknn_query(this->context,RKNN_QUERY_IN_OUT_NUM,&this->io_num,sizeof(this->io_num));
    if(ret < 0)
    {
        printf("rknn_query error ret=%d\n",ret);
    }
    printf("input num : %d, ouput num : %d.\n", io_num.n_input, io_num.n_output);
    // 获取输入和输出张量的属性
    input_attrs.resize(io_num.n_input);
    output_attrs.resize(io_num.n_output);
    /* 获取模型需要的输入和输出的tensor信息 */
    for(int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(this->context,RKNN_QUERY_INPUT_ATTR,&input_attrs[i],sizeof(input_attrs[i]));
        if(ret < 0)
        {
            printf("rknn_query error ret=%d\n",ret);
        }
    }
    for(int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(this->context,RKNN_QUERY_OUTPUT_ATTR,&output_attrs[i],sizeof(output_attrs[i]));
        if(ret < 0)
        {
            printf("rknn_query error ret=%d\n",ret);
        }
    }
    /* 获取模型要求输入图像的参数信息 */
    // 根据输入张量的格式确定模型的维度信息
    if(input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        model_channel = input_attrs[0].dims[0];
        model_height = input_attrs[0].dims[1];
        model_width = input_attrs[0].dims[2];
    }
    if(input_attrs[0].fmt == RKNN_TENSOR_NHWC)
    {
        model_channel = input_attrs[0].dims[3];
        model_height = input_attrs[0].dims[1];
        model_width = input_attrs[0].dims[2];
    }
}
yolov5s::~yolov5s()
{
    if(context)
    {
        rknn_destroy(context);
    }
    free(this->model_data);
}
unsigned char * yolov5s::load_model(const char *model_path,unsigned int &model_size)
{
    int ret;
    FILE *fp;
    unsigned char *model_data;
    //打开文件 获取句柄
    fp = fopen(model_path,"rb");
    if(fp == NULL)
    {
        printf("fopen %s fail!\n",model_path);
        return NULL;
    }
    //先从头找到尾，算出模型大小 ftell
    ret = fseek(fp,0,SEEK_END);
    if(ret)
    {
        printf("fseek fail!\n");
        return NULL;
    }

    model_size = ftell(fp);
    //分配内存
    model_data = (unsigned char *)malloc(model_size);
    //从开始处读载入模型数据
    ret = fseek(fp,0,SEEK_SET);
    if(ret)
    {
        printf("fseek fail!\n");
    }
    ret = fread(model_data,1,model_size,fp);
    if(ret < 0)
    {
        printf("fread fail!\n");
        free(model_data);
        return NULL;
    }
    return model_data;

}
int yolov5s::inference_image(const Mat &origin_img)
{
    int ret = 0;
    Mat img_cvt ,img_rga;
    int inputs_num,outputs_num;
    rknn_input inputs[1];
    rknn_output outputs[3];
    float scale_w,scale_h;
    vector<int32_t> qnt_zps;
    vector<float> qnt_scales;
    
    Mat bkg;
    memset(inputs,0,sizeof(inputs));
    memset(outputs,0,sizeof(outputs));
    this->img_channel = origin_img.channels();
    this->img_height = origin_img.rows;//获取图像的高度
    this->img_width = origin_img.cols;//获取图像的宽度
    printf("Image Height: %d\n", img_height);
    printf("Image Width: %d\n", img_width);
    printf("Image Channels: %d\n", img_channel);
    //检查图像尺寸是否为 16 的倍数
    if(this->img_width % 16 != 0 || this->img_height % 16 != 0)
    {
        int bkg_width = (img_width + 15)/ 16 * 16;    //计算宽度的16的倍数
        int bkg_height = (img_height + 15)/ 16 * 16;
        bkg = Mat(bkg_height,bkg_width,CV_8UC3,Scalar(0,0,0));//创建一个背景图片
        //copyTo
        origin_img.copyTo(bkg(Rect(0,0,origin_img.cols,origin_img.rows)));
        imwrite("img_bkg.jpg",bkg);
        this->img_height = bkg_height;
        this->img_width = bkg_width;
    }
    else
    {
        bkg = origin_img.clone();
    }
    int resize_width =  this->img_width;
    int resize_height = this->img_height;
    int resize_chanbel = this->img_channel; 
    

    printf("Resize Height: %d\n", resize_height);
    printf("Resize Width: %d\n", resize_width);
    printf("Resize Channels: %d\n", resize_chanbel);

    //对图像进行 rga 处理
    char * src_buf,*dst_buf,*src_cvt_buf;
    rga_buffer_handle_t src_handle,dst_handle,src_cvt_handle;
    //分配内存
    src_buf = (char *)malloc(img_width * img_height * img_channel);
    src_cvt_buf = (char *)malloc(img_width * img_height * img_channel);
    dst_buf = (char *)malloc(resize_width * resize_height * resize_chanbel);
    //复制数据并初始化内存
    memcpy(src_buf,bkg.data,img_width * img_height * img_channel);
    memset(src_cvt_buf,0x00,img_width * img_height * img_channel);
    memset(dst_buf,0x00,resize_width * resize_width * resize_chanbel);
    //导入缓冲区
    src_handle = importbuffer_virtualaddr(src_buf,img_width * img_height * img_channel);
    src_cvt_handle = importbuffer_virtualaddr(src_cvt_buf,img_width * img_height * img_channel);
    dst_handle = importbuffer_virtualaddr(dst_buf,resize_width * resize_height * resize_chanbel);
    
    if(src_handle == 0 || dst_handle == 0 || src_cvt_handle == 0)
    {
        printf("importbuffer_virtualaddr fail!\n");
        return -1;
    }
    //定义缓冲区
    rga_buffer_t src,dst,src_cvt;
    src = wrapbuffer_handle(src_handle,img_width,img_height,RK_FORMAT_BGR_888);
    src_cvt = wrapbuffer_handle(src_cvt_handle,img_width,img_height,RK_FORMAT_RGB_888);
    dst = wrapbuffer_handle(dst_handle,resize_width,resize_height,RK_FORMAT_RGB_888);
    // 检查图像格式
    ret = imcheck(src,dst,{},{});
    if(ret != IM_STATUS_NOERROR)
    {
        printf("imcheck error ret=%d\n",ret);
        goto end;
    }
    // 检查图像格式
    ret =imcvtcolor(src,src_cvt,RK_FORMAT_BGR_888,RK_FORMAT_RGB_888);
    if(ret != IM_STATUS_SUCCESS)
    {
        printf("imcvtcolor error ret=%d\n",ret);
        goto end;
    }
    // 调整图像大小
    ret = imresize(src_cvt,dst);
    if(ret != IM_STATUS_SUCCESS)
    {
        printf("imresize error ret=%d\n",ret);
        goto end;
    }
    img_cvt = Mat(img_height,img_width,CV_8UC3,src_cvt_buf);
    imwrite("img_cvt.jpg",img_cvt);
    img_rga = Mat(resize_height,resize_width,CV_8UC3,dst_buf);
    imwrite("img_rga.jpg",img_rga);
    // 推理 设置 rknn 输入参数
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].pass_through = false;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = dst_buf;
    //设置 rnkk 输出参数
    for(int i = 0; i < 3; ++i)
    {
        outputs[i].want_float = 1;
    }
    //运行 rknn
    ret = rknn_run(context,NULL);
    if(ret < 0)
    {
        printf("rknn_run error ret=%d\n",ret);
        return -1;
    }
    //获取模型输出
    rknn_outputs_get(context,3,outputs,NULL);
    return ret;

end:
    if(src_handle)
    {
        releasebuffer_handle(src_handle);
    
    }

    if(dst_handle)
    {
        releasebuffer_handle(dst_handle);
    
    }
    if(src_cvt_handle)
    {
        releasebuffer_handle(src_cvt_handle);
    
    }
}