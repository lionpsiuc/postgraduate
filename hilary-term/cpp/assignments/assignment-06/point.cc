/**
 * @file  point.cc
 * @brief Implementation of Point class and related functions.
 */

#include <algorithm>
#include <fstream>
#include <iostream>

#include "point.h"

#ifdef MPI
#include <boost/mpi/packed_iarchive.hpp>
#include <boost/mpi/packed_oarchive.hpp>
#include <boost/serialization/export.hpp>
#endif

/**
 * Equality operator for Point class. Compares x- and y-coordinates of two
 * points.
 *
 * @param rhs The right-hand side point to compare with.
 * @returns   True if points have identical coordinates, false otherwise.
 */
bool Point::operator==(const Point& rhs) {
  return (x == rhs.x) && (y == rhs.y);
}

/**
 * Writes a vector of points to a specified file, ensuring the polygon is
 * closed.
 *
 * @param fn  The filename.
 * @param pts The vector of points to write. The function will append the first
 *            point to the end of this (copied) vector to close the polygon for
 *            plotting.
 */
void write_to_file(std::string fn, std::vector<Point> pts) { // pts is a copy
  std::ofstream outfile(fn);
  if (!outfile) {
    std::cerr << "Error: Could not open file " << fn << std::endl;
    return;
  }

  // If there are points, add the first point to the end of the vector
  if (!pts.empty()) {
    pts.push_back(pts.front());
  }

  for (const auto& pt : pts) {
    outfile << pt << '\n';
  }
  outfile.close();
}

/**
 * Calculates cross product to determine the orientation of three points.
 *
 * @param p1 First point (origin for the two vectors).
 * @param p2 Second point (defines the first vector).
 * @param p3 Third point (defines the second vector).
 * @returns  True if the cross product (p2-p1) x (p3-p1) is negative or zero.
 */
bool cross_prod(Point p1, Point p2, Point p3) {
  double cross = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
  return cross <= 0;
}

/**
 * Sorts points by x-coordinate in ascending order; however, if x-coordinates
 * are equal, then they will be sorted by the y-coordinate instead.
 *
 * @param points Vector of points to be sorted.
 */
void sort_points(std::vector<Point>& points) {
  std::sort(points.begin(), points.end(), [](const Point& a, const Point& b) {
    if (a.x == b.x) {
      return a.y <
             b.y; // Secondary sort by y-coordinate when x-coordinates are equal
    }
    return a.x < b.x; // Primary sort by x-coordinate
  });
}

#ifdef MPI
template void Point::serialize<boost::mpi::packed_oarchive>(
    boost::mpi::packed_oarchive& ar, unsigned version);
template void Point::serialize<boost::mpi::packed_iarchive>(
    boost::mpi::packed_iarchive& ar, unsigned version);
#endif
