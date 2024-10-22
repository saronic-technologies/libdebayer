#include "threadpool.hpp"
#include <iostream>
#include <thread>

#ifdef __linux__
    #include <pthread.h>    // For pthread_setaffinity_np
    #include <sched.h>      // For CPU_SET and CPU_ZERO
#endif

// Constructor Implementation
ThreadPool::ThreadPool(size_t num_threads)
    : stop(false), tasks_in_progress(0)
{
    if (num_threads == 0)
        throw std::invalid_argument("Number of threads must be greater than zero.");

    size_t num_cores = std::thread::hardware_concurrency();
    if (num_cores == 0)
        num_cores = 1; // Fallback to at least one core

    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back(&ThreadPool::WorkerThread, this);

#ifdef __linux__
        // Linux-specific thread affinity
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i % num_cores, &cpuset); // Distribute threads across available cores

        int rc = pthread_setaffinity_np(workers.back().native_handle(),
                                        sizeof(cpu_set_t), &cpuset);
        if (rc != 0) {
            std::cerr << "Error setting thread affinity on Linux: " << rc << std::endl;
        }
#endif
    }
}

// Destructor Implementation
ThreadPool::~ThreadPool()
{
    // Signal all threads to stop
    stop.store(true, std::memory_order_relaxed);
    condition.notify_all();

    // Join all threads
    for (std::thread &worker : workers) {
        if (worker.joinable())
            worker.join();
    }
}

// Submit a task to the thread pool
void ThreadPool::Submit(const std::function<void()>& task)
{
    {
        // Increment the task counter
        tasks_in_progress.fetch_add(1, std::memory_order_relaxed);

        // Enqueue the task
        tasks.enqueue(task);
    }
    // Notify one worker
    condition.notify_one();
}

// Wait for all submitted tasks to complete
void ThreadPool::WaitAll()
{
    std::unique_lock<std::mutex> lock(wait_mutex);
    wait_condition.wait(lock, [this]() { 
        return tasks_in_progress.load(std::memory_order_relaxed) == 0 && tasks.size_approx() == 0; 
    });
}

// Worker thread function
void ThreadPool::WorkerThread()
{
    while (true) {
        std::function<void()> task;
        bool dequeued = false;

        // Attempt to dequeue a task
        dequeued = tasks.try_dequeue(task);

        if (dequeued) {
            // Execute the task
            try {
                task();
            } catch (const std::exception& e) {
                std::cerr << "Exception in task: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "Unknown exception in task." << std::endl;
            }

            // Decrement the task counter
            if (tasks_in_progress.fetch_sub(1, std::memory_order_relaxed) == 1) {
                // If this was the last task, notify waiters
                std::lock_guard<std::mutex> lock(wait_mutex);
                wait_condition.notify_all();
            }
        } else {
            // If no task is available, wait for notification
            std::unique_lock<std::mutex> lock(condition_mutex);
            condition.wait(lock, [this]() { 
                return this->stop.load(std::memory_order_relaxed) || this->tasks.size_approx() > 0;
            });

            if (this->stop.load(std::memory_order_relaxed) && this->tasks.size_approx() == 0)
                return;
        }
    }
}
