#include "threadpool.h"
threadpool::threadpool(int thread_num)
{
    run_falg = true;
    Init(thread_num);
}
threadpool::~threadpool()
{
    run_falg = false;
    task_cond.notify_all();
    for(auto &t : threads)
    {
        if(t.joinable())
        {
            t.join();
        }
    }
    cout<<"threadpool destory!.\n"<<endl;
}
void threadpool::Init(int thread_num)
{ 
    if(thread_num <= 0)
    {
        thread_num = 1;
    }
    //创建线程
    for(int i = 0; i < thread_num; i++ )
    {
        //循环thread_num次创建线程
        //每个线程都执行threadpool::worker函数
        //this指针传递当前线程池对象实例
        //i作为线程ID参数传递
        //使用emplace_back将新创建的线程对象直接构造到threads容器中
        threads.emplace_back(&threadpool::worker , this, i);
    }
}
void threadpool::worker(int worker_id)
{
    cout << "threadpool::worker " << worker_id << " start" << endl;
    while (run_falg)
    {
        packaged_task<ProcessResult()> curr_task;
        {
            unique_lock<mutex> lock(task_mutex);
            task_cond.wait(lock,[this] {return !tasks.empty() || !run_falg;});//队列非空或者 no run
            if(!run_falg)
            {
                std::cout << "worker " << worker_id << " 下班！\n";
                break;
            }
            //从 tasks 队列取出一个打包后的任务
            curr_task = move(tasks.front());
            tasks.pop();
        }
        //离开大锁区开始执行真正的推理任务
        if(curr_task.valid())
        {
            printf("woker %d start task.\n",worker_id);
            curr_task();
        } 

    }
    // 在worker线程退出时添加
std::cout << "Worker " << worker_id << " exited, remaining tasks: " << tasks.size() << std::endl;
}
future<ProcessResult>  threadpool::submit_task_async(int index,cv::Mat img)    
{
    packaged_task<ProcessResult()> task([this,index,img]()
    {
        ProcessResult result;
        try
        {
            
            result.process_img = img;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        return result;
    });
    
    future<ProcessResult> future = task.get_future();
    {
        unique_lock<mutex> lock(task_mutex);
        //把打包好的任务放入队列
        tasks.emplace(move(task));
        std::cout << "[submit_task_async] 已压入tasks队列, 现在大小=" << tasks.size() << std::endl;
    }
    task_cond.notify_one();
    return future;
}
