#include "threadpool.hpp"
#include <iostream>
#include <algorithm>


// Constructor Implementation
ThreadPool::ThreadPool(size_t num_threads) : stop(false) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back([this, i]() {
            while (true) {
                std::function<void()> task;
                bool dequeued = false;

                // Attempt to dequeue a task
                dequeued = tasks.try_dequeue(task);

                if (dequeued) {
                    task();
                } else {
                    // If no task is available, wait for notification
                    std::unique_lock<std::mutex> lock(this->condition_mutex);
                    this->condition.wait(lock, [this]() { 
                        return this->stop.load() || this->tasks.size_approx() != 0; 
                    });

                    if (this->stop.load() && this->tasks.size_approx() <= 0)
                        return;
                }
            }
        });
    }
}

// Destructor Implementation
ThreadPool::~ThreadPool() {
    stop.store(true);
    condition.notify_all();
    for (std::thread &worker : workers)
        if (worker.joinable())
            worker.join();
}
