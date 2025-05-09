/**
 * @file  assignment6a.cc
 * @brief Parallelizing std::inner_product.
 */

#include <algorithm>
#include <chrono>
#include <functional>
#include <future>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

/**
 * Launch policy for asynchronous tasks.
 *
 * Controls whether tasks run concurrently with the caller or are deferred until
 * explicitly requested. Set to async for immediate concurrent execution.
 */
const auto policy = std::launch::async;
// const auto policy = std::launch::deferred;

/**
 * Compares different implementations of parallel dot product calculations.
 *
 * Generates two random vectors of doubles and calculates their dot product
 * (inner product) using three different methods:
 *
 *   1. Serial computation with std::inner_product.
 *   2. Parallel computation with std::async.
 *   3. Parallel computation with std::packaged_task and std::thread.
 *
 * Times each approach and prints the results to standard output.
 *
 * @returns 0 upon successful completion.
 */
int main(void) {
  const int           n{50'000'000};
  std::atomic<double> dot_prod{0};

  // Random number generation
  std::random_device         rd{};
  std::default_random_engine eng{rd()};
  auto ui = std::bind(std::normal_distribution<>{}, std::ref(eng));

  // Define vectors to use for the calculations
  std::vector<double> v1(n);
  std::vector<double> v2(n);

  // Fill them with random values
  std::generate(std::begin(v1), std::end(v1), ui);
  std::generate(std::begin(v2), std::end(v2), ui);

  auto hardware_threads = std::thread::hardware_concurrency();
  std::cout << "This system has " << hardware_threads << " threads\n";

  // Determine number of threads for computations
  unsigned int thread_count = std::max(1u, hardware_threads);
  std::cout << "Using " << thread_count
            << " threads for parallel computation\n\n";

  /**
   * Calculates partial dot product between segments of two vectors.
   *
   * @param   it  Iterator to the beginning of the first vector segment.
   * @param   it2 Iterator to the end of the first vector segment.
   * @param   it3 Iterator to the beginning of the second vector segment.
   * @returns     Dot product of the specified segments.
   */
  auto partial_dot = [](auto it, auto it2, auto it3) {
    return std::inner_product(it, it2, it3, 0.0);
  };

  /*
   * Serial
   */
  auto start = std::chrono::steady_clock::now();
  dot_prod =
      std::inner_product(std::begin(v1), std::end(v1), std::begin(v2), 0.0);
  auto end = std::chrono::steady_clock::now();
  std::cout << "Serial dot_prod = " << dot_prod << "\nElapsed time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms\n\n";

  /*
   * std::async
   */
  start = std::chrono::steady_clock::now();

  // Calculate segment size based on the number of threads
  size_t segment_size = n / thread_count;

  // Vector to hold futures for each task
  std::vector<std::future<double>> futures;
  futures.reserve(thread_count);

  // Launch std::async tasks for each segment
  for (unsigned int i = 0; i < thread_count; i++) {
    auto start_it1 = std::begin(v1) + i * segment_size;
    auto start_it2 = std::begin(v2) + i * segment_size;

    // For the last thread, process until the end of the vector
    auto end_it1 =
        (i == thread_count - 1) ? std::end(v1) : start_it1 + segment_size;

    // Launch the std::async task and store its future
    futures.push_back(
        std::async(policy, partial_dot, start_it1, end_it1, start_it2));
  }

  // Collect results from all futures
  double temp_result = 0.0;
  for (auto& future : futures) {
    temp_result += future.get();
  }
  dot_prod = temp_result;

  end = std::chrono::steady_clock::now();
  std::cout << "std::async dot_prod = " << dot_prod << "\nElapsed time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms\n\n";

  /*
   * std::packaged_task and std::thread
   */
  start = std::chrono::steady_clock::now();

  // Create vectors to store tasks, futures, and threads, respectively
  std::vector<std::packaged_task<double()>> tasks(thread_count);
  std::vector<std::future<double>>          task_futures(thread_count);
  std::vector<std::thread>                  threads(thread_count);

  // Set up tasks for each thread
  for (unsigned int i = 0; i < thread_count; i++) {
    auto start_it1 = std::begin(v1) + i * segment_size;
    auto start_it2 = std::begin(v2) + i * segment_size;

    // For the last thread, process until the end of the vector
    auto end_it1 =
        (i == thread_count - 1) ? std::end(v1) : start_it1 + segment_size;

    // Create std::packaged_task using the lambda expression
    tasks[i] = std::packaged_task<double()>([=, &partial_dot]() {
      return partial_dot(start_it1, end_it1, start_it2);
    });

    // Get future from the task
    task_futures[i] = tasks[i].get_future();

    // Launch thread with the task
    threads[i] = std::thread(std::move(tasks[i]));
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // Collect results from all futures
  double dp = 0.0;
  for (auto& future : task_futures) {
    dp += future.get();
  }

  end = std::chrono::steady_clock::now();
  std::cout << "std::packaged_task and std::thread dot_prod = " << dp
            << "\nElapsed time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms\n\n";

  return 0;
}
