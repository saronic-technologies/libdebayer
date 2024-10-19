// ThreadPool.hpp
#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>
#include <stdexcept>
#include <memory>
#include <iostream>

#ifdef __linux__
#include <pthread.h>
#include <unistd.h>
#endif

class ThreadPool {
public:
    // Constructor: Initializes the thread pool with a given number of threads
    ThreadPool(size_t num_threads);

    // Destructor: Waits for all tasks to finish and joins all threads
    ~ThreadPool();

    // Submit a task to the thread pool
    template<class F, class... Args>
    auto Submit(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task_ptr = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task_ptr->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            // Don't allow enqueueing after stopping the pool
            if (stop.load())
                throw std::runtime_error("Submit on stopped ThreadPool");

            tasks.emplace([task_ptr](){ (*task_ptr)(); });
        }
        condition.notify_one();
        return res;
    }

    // Prevent copy and assignment
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

private:
    // Worker threads
    std::vector<std::thread> workers;

    // Task queue
    std::queue<std::function<void()>> tasks;

    // Synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;

    // Pin a thread to a specific CPU core
    void PinThreadToCore(std::thread& thread, size_t core_id);
};

#endif // THREAD_POOL_HPP
