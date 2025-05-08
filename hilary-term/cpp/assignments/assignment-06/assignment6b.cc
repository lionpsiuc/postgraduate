/**
 * @file  assignment6b.cc
 * @brief Main function demonstrating the ConvexHull class.
 */

#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "ConvexHull.h"
#include "point.h"

// Use below two lines to seed from random device
std::random_device         rd;
std::default_random_engine rng{rd()};

// Use the below with some seed for a fixed seed
// std::default_random_engine rng {1234};

const auto policy = std::launch::async;
// const auto policy = std::launch::deferred;

int main(void) {

  // You can change number of points if you want for timing purposes
  const int          n{100};
  std::vector<Point> points(n);

  auto gaussian_rv = std::bind(std::normal_distribution<>{0, 1}, std::ref(rng));

  // Different distribution
  // auto gaussian_rv = std::bind(std::uniform_real_distribution<> {0,1},
  // std::ref(rng)); // If you want a different distribution

  // Assign random coordinates to each point
  for (auto& p : points) {
    p.x = gaussian_rv();
    p.y = gaussian_rv();
  }

  // Sort in order of increasing x-coordinates
  sort_points(points);

  // Construct a convex hull from these point
  ConvexHull CH{std::begin(points), std::end(points)};

  auto start = std::chrono::steady_clock::now();

  CH.generate_hull(); // Generate hull

  auto end = std::chrono::steady_clock::now();

  std::cout << "Serial time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms\n\n";

  write_to_file(std::string("points.txt"),
                CH.get_points()); // Write all points to file
  write_to_file(std::string("serial.txt"),
                CH.get_hull()); // Write hull points to file

  /*
   * You don't need to change above this line
   */

  // Determine the size of each segment
  size_t total_points = points.size();
  size_t segment_size = total_points / 3;

  // Create iterators to the beginning, boundaries, and end of segments
  auto points_begin = points.begin();
  auto points_mid1  = points_begin + segment_size;
  auto points_mid2  = points_mid1 + segment_size;
  auto points_end   = points.end();

  // Create the three ConvexHull objects for each third of the points
  ConvexHull CH_left(points_begin, points_mid1);
  ConvexHull CH_mid(points_mid1, points_mid2);
  ConvexHull CH_right(points_mid2, points_end);

  start = std::chrono::steady_clock::now();
  {

    // Launch std::async tasks to generate the hulls in parallel; store the
    // futures to avoid blocking until task completion
    auto fut_left  = std::async(policy, &ConvexHull::generate_hull, &CH_left);
    auto fut_mid   = std::async(policy, &ConvexHull::generate_hull, &CH_mid);
    auto fut_right = std::async(policy, &ConvexHull::generate_hull, &CH_right);

    // If using std::launch::deferred, we need to wait for or get the results
    if (policy == std::launch::deferred) {
      fut_left.wait();
      fut_mid.wait();
      fut_right.wait();
    }
  }

  /*
   * You don't need to change below this line
   */

  // If you want to be more accurate, comment out the write_to_file functions
  // when timing the code
  write_to_file(std::string("left_hull.txt"), CH_left.get_hull());
  write_to_file(std::string("mid_hull.txt"), CH_mid.get_hull());
  write_to_file(std::string("right_hull.txt"), CH_right.get_hull());

  CH_left.merge_to_right(CH_mid); // Merge first two hulls into CH_left
  write_to_file(std::string("after_first_merge.txt"), CH_left.get_hull());

  CH_left.merge_to_right(CH_right); // Merge other hull into CH_right
  write_to_file(std::string("after_second_merge.txt"), CH_left.get_hull());

  end = std::chrono::steady_clock::now();
  std::cout << "Parallelised time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms\n\n";

  return 0;
}
