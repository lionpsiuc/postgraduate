/**
 * @file  point.cc
 * @brief Implementation of Point class and related functions.
 */

#include <algorithm>
#include <fstream>
#include <iostream>

#include "point.h"

/**
 * @brief     Equality operator for Point class. Compares x and y coordinates of
 *            two points.
 * @param rhs The right-hand side point to compare with.
 * @return    True if points have identical coordinates, false otherwise.
 */
bool Point::operator==(const Point& rhs) {
  return (x == rhs.x) && (y == rhs.y);
}

/**
 * @brief     Writes a vector of points to a file. Each point is written on a
 *            new line with the format defined by the Point's << operator in
 *            point.h.
 * @param fn  The filename to write to.
 * @param pts Vector of points to write.
 */
void write_to_file(std::string fn, std::vector<Point> pts) {
  std::ofstream outfile(fn);
  if (!outfile) {
    std::cerr << "Error: Could not open file " << fn << std::endl;
    return;
  }
  for (const auto& pt : pts) {
    outfile << pt << '\n';
  }
  outfile.close();
}

/**
 * @brief    Calculates cross product to determine the orientation of three
 *           points.
 * @param p1 First point
 * @param p2 Second point
 * @param p3 Third point
 * @return   True if p3 is to the left of the line from p1 to p2 (i.e.,
 *           counterclockwise turn)
 */
bool cross_prod(Point p1, Point p2, Point p3) {
  double cross = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
  return cross > 0;
}

/**
 * @brief        Sorts points by x-coordinate in ascending order; however, if
 *               x-coordinates are equal, then they will be sorted by the
 *               y-coordinate instead.
 * @param points Vector of points to be sorted
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
