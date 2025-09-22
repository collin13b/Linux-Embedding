#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <thread>
#include <queue>
#include <atomic>
#include <chrono>
#include "threadpool.h"   
#include "/home/cat/linuxAi/rk3588prj/linuxlLearning/SafeQueue.h"
using namespace std;
using namespace cv;
struct FrameData
{
    Mat img;
    int frame_id;
};
SafeQueue<FrameData> read_queue(400);
SafeQueue<FrameData> write_queue(400);
atomic<bool> read_finished (false);
atomic<bool> process_finished (false);
void ReadThread(VideoCapture &cap)
{
    FrameData frame_temp;
    int index = 0;
    cv::Mat img;
    while (1)
    {
        
        
        //读不到帧
        if(!cap.read(frame_temp.img))
        {
            printf("read frame failed.\n");
            break;
        }
        frame_temp.img = img;
        frame_temp.frame_id = index;
        index++;
        //入队
        read_queue.enqueue(frame_temp);
        printf("read frame %d.\n",index);
        
    }
    read_finished = true;
    printf("read finish.\n");
}
void ProcessThread(threadpool &nup_pool)
{
    while(1)
    {
        if(!read_queue.empty())
        {
            FrameData frame_temp;;
            int idx = 0;
            read_queue.dequeue(frame_temp);
            Mat img = frame_temp.img;   
            write_queue.enqueue({move(img),idx++});
            printf("process frame %d.\n",idx);
            idx++;
        }
        else if(read_finished)
        {
        break;
        }
    }
    process_finished = true;
    printf("process finish.\n");
}
void WriteThread(VideoWriter &writer)
{   int idx = 0;
    Mat img;
    while(1)
    {
        if(!write_queue.empty())
        {
            FrameData frame_temp;
            write_queue.dequeue(frame_temp);
            img = frame_temp.img;
            writer.write(img);
            idx++;
            printf("write frame %d.\n",idx);
        }
        else if(process_finished && read_queue.empty())
        {
        break;
        }
        
    }
    
    printf("write finish.\n");
}
int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    const string INPUT_VIDEO_PATH = "/home/cat/linuxAi/rk3588prj/linuxlLearning/video.mp4";
    const string OUTPUT_VIDEO_PATH = "/home/cat/linuxAi/rk3588prj/linuxlLearning/output.avi";
    VideoCapture cap(INPUT_VIDEO_PATH);
    if(!cap.isOpened())
    {
        printf("open video failed.\n");
        return -1;
    }
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);
    int frame_count = cap.get(CAP_PROP_FRAME_COUNT);
    int fourcc = cv::VideoWriter::fourcc('I', '4', '2', '0');
    threadpool nup_pool(4);
    //输出视频
    VideoWriter writer(OUTPUT_VIDEO_PATH,fourcc,fps,Size(width,height));
    if(!writer.isOpened())
    {
        printf("open video failed.\n");
        return -1;
    }  

    thread read_thread(ReadThread,ref(cap));
    thread process_thread(ProcessThread,ref(nup_pool));
    thread write_thread(WriteThread,ref(writer));
    while (1)
    {
        /* code */
    }
    
    
    
    read_thread.join();
    process_thread.join();
    write_thread.join();



    read_queue.stop();
    write_queue.stop();

    writer.release();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<chrono::milliseconds>(end - start);
    printf("time cost %d ms.\n",duration.count());
    printf("all done.\n");
    return 0;
}