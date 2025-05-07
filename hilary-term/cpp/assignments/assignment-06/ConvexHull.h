/**
 * @file  ConvexHull.h
 * @brief Contains class definition for ConvexHull.
 */

#ifndef CONVEXHULL_H
#define CONVEXHULL_H

#include "point.h"

class ConvexHull {
public:
  ConvexHull(std::vector<Point>::iterator beg,
             std::vector<Point>::iterator end);
  ConvexHull()                                    = default;
  ~ConvexHull()                                   = default; // Destructor
  ConvexHull(const ConvexHull&)                   = default; // Copy constructor
  ConvexHull(ConvexHull&&)                        = default; // Move constructor
  ConvexHull&        operator=(const ConvexHull&) = default; // Copy assignment
  ConvexHull&        operator=(ConvexHull&&)      = default; // Move assignment
  void               merge_to_right(ConvexHull& right);
  std::vector<Point> generate_hull();
  std::vector<Point> get_points() const { return points; }
  std::vector<Point> get_hull() const { return hull; }

private:
  std::vector<Point> points;
  std::vector<Point> hull;
};

#endif
