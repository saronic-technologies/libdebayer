// ThreadPool.cpp
#include "threadpool.hpp"

ThreadPool::ThreadPool(size_t num_threads) : stop(false) {
    size_t physical_cores = std::thread::hardware_concurrency();
    size_t threads_to_create = std::min(num_threads, physical_cores);

    for (size_t i = 0; i < threads_to_create; ++i) {
        workers.emplace_back([this, i]() {
            // Pin the thread to a specific core
#ifdef __linux__
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i % physical_cores, &cpuset);
            int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            if (rc != 0) {
                std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
            }
#endif
            while (true) {
                std::function<void()> task;

                {   // Acquire lock
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, 
                        [this]{ return this->stop.load() || !this->tasks.empty(); });
                    if (this->stop.load() && this->tasks.empty())
                        return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }   // Release lock

                // Execute the task
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    stop.store(true);
    condition.notify_all();
    for (std::thread &worker: workers)
        if (worker.joinable())
            worker.join();
}

void ThreadPool::PinThreadToCore(std::thread& thread, size_t core_id) {
    // Implementation if needed (already handled in constructor lambda)
}
