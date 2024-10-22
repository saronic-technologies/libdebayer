// ThreadPool.hpp
#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <vector>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <stdexcept>
#include <memory>
#include <iostream>

// Include moodycamel's ConcurrentQueue
#include "concurrentqueue.h"

#ifdef __linux__
#include <pthread.h>
#include <unistd.h>
#endif

class ThreadPool {
public:
    // Constructor: Initializes the thread pool with a given number of threads
    explicit ThreadPool(size_t num_threads);

    // Destructor: Waits for all tasks to finish and joins all threads
    ~ThreadPool();

    // Submit a task to the thread pool
    void Submit(const std::function<void()>& task);

    // Wait for all submitted tasks to complete
    void WaitAll();

    // Prevent copy and assignment
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

private:
    // Worker threads
    std::vector<std::thread> workers;

    // Task queue using moodycamel's ConcurrentQueue
    moodycamel::ConcurrentQueue<std::function<void()>> tasks;

    // Synchronization for condition variable
    std::mutex condition_mutex;
    std::condition_variable condition;

    // Synchronization for waiting
    std::mutex wait_mutex;
    std::condition_variable wait_condition;

    // Atomic flag to stop the pool
    std::atomic<bool> stop;

    // Atomic counter for tasks in progress
    std::atomic<size_t> tasks_in_progress;

    // Worker thread function
    void WorkerThread();
};

#endif // THREAD_POOL_HPP
