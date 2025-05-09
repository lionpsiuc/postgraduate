/**
 * @file  assignment6c.cc
 * @brief Main function demonstrating the ConvexHull class using Boost.MPI.
 */

#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>

#include "ConvexHull.h"
#include "point.h"

// Use below two lines to seed from random device
std::random_device         rd;
std::default_random_engine rng{rd()};

// Use the below with some seed for a fixed seed
// std::default_random_engine rng {1234};

int main(int argc, char* argv[]) {
  // Initialize MPI environment
  boost::mpi::environment  env(argc, argv);
  boost::mpi::communicator world;

  // Number of points to generate
  const int          n{10000000};
  std::vector<Point> points;
  std::vector<Point> local_points;

  // Timer variables for measuring performance
  std::chrono::time_point<std::chrono::steady_clock> start, end;

  // Process 0 generates all points and calculates serial hull for comparison
  if (world.rank() == 0) {
    std::cout << "Running with " << world.size() << " MPI processes\n";
    points.resize(n);

    // Initialize a Gaussian random variable generator
    auto gaussian_rv =
        std::bind(std::normal_distribution<>{0, 1}, std::ref(rng));

    // Assign random coordinates to each point
    for (auto& p : points) {
      p.x = gaussian_rv();
      p.y = gaussian_rv();
    }

    // Sort in order of increasing x-coordinates
    sort_points(points);

    // Write all points to file for visualization
    // write_to_file(std::string("points.txt"), points);

    // Construct a convex hull from these points (serial calculation)
    ConvexHull CH{std::begin(points), std::end(points)};

    start = std::chrono::steady_clock::now();

    CH.generate_hull(); // Generate hull serially

    end = std::chrono::steady_clock::now();

    std::cout << "Serial: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << " ms\n\n";

    // Write hull points to file for visualization
    // write_to_file(std::string("serial.txt"), CH.get_hull());

    // Start timing the parallel version
    start = std::chrono::steady_clock::now();

    // Calculate segment size based on number of processes (using 3 processes)
    size_t segment_size = points.size() / 3;

    // Keep a portion of the points for process 0
    local_points.assign(points.begin(), points.begin() + segment_size);

    // Send remaining points to processes 1 and 2
    if (world.size() > 1) {
      world.send(1, 0,
                 std::vector<Point>(points.begin() + segment_size,
                                    points.begin() + 2 * segment_size));
    }

    if (world.size() > 2) {
      world.send(
          2, 0,
          std::vector<Point>(points.begin() + 2 * segment_size, points.end()));
    }
  } else {
    // Processes 1 and 2 receive their portion of points
    world.recv(0, 0, local_points);
  }

  // Each process calculates the convex hull for its portion of points
  ConvexHull local_CH(local_points.begin(), local_points.end());
  local_CH.generate_hull();

  // For visualization, have each process write its hull to a file
  // if (world.rank() == 0) {
  //   write_to_file(std::string("left_hull.txt"), local_CH.get_hull());
  // } else if (world.rank() == 1 && local_CH.get_hull().size() > 0) {
  //   write_to_file(std::string("mid_hull.txt"), local_CH.get_hull());
  // } else if (world.rank() == 2 && local_CH.get_hull().size() > 0) {
  //   write_to_file(std::string("right_hull.txt"), local_CH.get_hull());
  // }

  // Processes 1 and 2 send their hulls back to process 0
  if (world.rank() != 0) {
    world.send(0, world.rank(), local_CH.get_hull());
  } else {
    // Process 0 receives hulls from processes 1 and 2 and merges them
    std::vector<Point> hull_from_1, hull_from_2;

    if (world.size() > 1) {
      world.recv(1, 1, hull_from_1);

      // Create a ConvexHull object for the received hull
      ConvexHull CH_mid;
      CH_mid.move_hull(std::move(hull_from_1));

      // Merge hulls
      local_CH.merge_to_right(CH_mid);

      // Write the intermediate result for visualization
      // write_to_file(std::string("after_first_merge.txt"),
      // local_CH.get_hull());
    }

    if (world.size() > 2) {
      world.recv(2, 2, hull_from_2);

      // Create a ConvexHull object for the received hull
      ConvexHull CH_right;
      CH_right.move_hull(std::move(hull_from_2));

      // Merge hulls
      local_CH.merge_to_right(CH_right);

      // Write the final result for visualization
      // write_to_file(std::string("after_second_merge.txt"),
      // local_CH.get_hull());
    }

    // End timing
    end = std::chrono::steady_clock::now();

    std::cout << "Parallelized: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << " ms\n\n";
  }

  return 0;
}
