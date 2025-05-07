/**
 * @file  point.h
 * @brief Class definition for Point class.
 */

#ifndef POINT_H
#define POINT_H

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <vector>

/**
 * @brief Simple structure to store coordinates of a point on the grid.
 */
struct Point {
  double x; // x-coordinate
  double y; // y-coordinate

  // Constructors
  Point() = default;
  Point(double inx, double iny) : x{inx}, y{iny} {};

  bool operator==(const Point& rhs);
};

/**
 * @brief    Overloading output stream operator for a Point. The operator should
 *           simply print the x and y coordinates of the Point in.
 * @param os Reference to output stream
 * @param in Reference to Point which we want to print.
 * @return   Reference to output stream
 */
inline std::ostream& operator<<(std::ostream& os, const Point& in) {
  os << std::fixed << std::setw(9) << in.x << ", " << std::setw(9) << in.y;
  return os;
}

void write_to_file(std::string fn, std::vector<Point> pts);
bool cross_prod(Point p1, Point p2, Point p3);
void sort_points(std::vector<Point>& points);

#endif
