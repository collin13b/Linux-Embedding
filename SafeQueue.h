#ifndef __SAFEQUEUE_H
#define __SAFEQUEUE_H
#include <queue>
#include <mutex>
#include <iostream>    
#include <condition_variable>
using namespace std;
template <typename T>
class SafeQueue{
public:
    SafeQueue(size_t max_size_in): maxSize(max_size_in){};//初始化变量
    ~SafeQueue(){};
    void enqueue(const T &t)//传入图像
    {
        unique_lock<mutex> lock(m);
        c_not_full.wait(lock,[this]{return q.size() < maxSize;});
        q.push(t);
     
        c_not_full.notify_one();
    }
    bool dequeue(T &t)//出队
    {
        unique_lock<mutex> lock(m);
        c_not_empty.wait(lock,[this]{return !q.empty();});
        t = q.front();
        q.pop();
        c_not_empty.notify_one();
        return true;
    }
    void stop()//暂停队列
    {
        lock_guard<mutex> lock(m);
        stop_flag = true;
        c_not_full.notify_all();
        c_not_empty.notify_all();
    }
    bool empty()//空
    {
        lock_guard<mutex> lock(m);
        return q.empty();
    }
    size_t size()//大小
    {
        lock_guard<mutex> lock(m);
        return q.size();    
    }

private:
    std::queue<T> q;
    std::mutex m;
    std::condition_variable c_not_empty;
    std::condition_variable c_not_full;
    bool stop_flag = false;
    size_t maxSize;
};
#endif