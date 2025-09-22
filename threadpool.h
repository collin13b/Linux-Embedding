#ifndef __THREADPOOL_H__
#define __THREADPOOL_H__ 
#include <atomic>
#include <thread>
#include <mutex>  
#include <vector>  
#include <queue>
#include<condition_variable>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <map>
#include <iostream>
#include <future>
using namespace std;
struct ProcessResult
{
    cv::Mat process_img;
};
class threadpool
{
public:
    threadpool(int thread_num);
    ~threadpool();
    future<ProcessResult> submit_task_async(int index,cv::Mat img);
private:
    void Init(int thread_num);
    void worker(int worker_id);
    //核心人物队列
    queue<packaged_task<ProcessResult()>> tasks;
    mutex task_mutex;
    condition_variable task_cond;   
    //线程池线程
    vector<thread> threads;
    atomic<bool> run_falg{true};
};
#endif  