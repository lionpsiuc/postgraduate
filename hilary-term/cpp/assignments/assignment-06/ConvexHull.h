/**
 * @file  ConvexHull.h
 * @brief Contains class definition for ConvexHull.
 */

#ifndef CONVEXHULL_H
#define CONVEXHULL_H

#include "point.h"

/**
 * Computes and represents the convex hull of a set of points.
 *
 * This class implements Andrew's monotone chain algorithm for generating the
 * hull.
 *
 * The class also supports merging convex hulls.
 *
 * @see Point
 */
class ConvexHull {
public:
  /**
   * Constructor that initializes the points from a range defined by iterators.
   *
   * @param beg Iterator to the beginning of a Point vector.
   * @param end Iterator to the end of a Point vector.
   */
  ConvexHull(std::vector<Point>::iterator beg,
             std::vector<Point>::iterator end);

  /**
   * Default constructor that creates an empty convex hull.
   */
  ConvexHull() = default;

  /**
   * Default destructor.
   */
  ~ConvexHull() = default;

  /**
   * Default copy constructor.
   */
  ConvexHull(const ConvexHull&) = default;

  /**
   * Default move constructor.
   */
  ConvexHull(ConvexHull&&) = default;

  /**
   * Default copy assignment operator.
   *
   * @returns Reference to this ConvexHull after assignment.
   */
  ConvexHull& operator=(const ConvexHull&) = default;

  /**
   * Default move assignment operator.
   *
   * @returns Reference to this ConvexHull after assignment.
   */
  ConvexHull& operator=(ConvexHull&&) = default;

  /**
   * Merges this convex hull with another hull on its right.
   *
   * This modifies the current hull to become the convex hull of the union of
   * points from both hulls. The algorithm finds upper and lower tangent lines
   * between the hulls and uses them to determine which points to keep.
   *
   * @param right The ConvexHull to merge with, which should generally contain
   *              points that are to the right of this hull's points.
   */
  void merge_to_right(ConvexHull& right);

  /**
   * Generates the convex hull using Andrew's monotone chain algorithm.
   *
   * This method computes the convex hull of the points stored in this object.
   * It assumes the points are already sorted by x-coordinate (and by
   * y-coordinate when x-coordinates are equal). If the points are not sorted,
   * the result will not be correct.
   *
   * The algorithm works by building the lower and upper hulls separately and
   * then combining them.
   *
   * @returns A vector of Points representing the vertices of the convex hull,
   *          in counterclockwise order.
   * @note    If there are fewer than three points, the hull will be the points
   *          themselves.
   */
  std::vector<Point> generate_hull();

  /**
   * Returns the vector of input points.
   *
   * @returns A copy of the vector containing all input points.
   */
  std::vector<Point> get_points() const { return points; }

  /**
   * Returns the computed convex hull vertices.
   *
   * @returns A copy of the vector containing the convex hull vertices. If
   *          generate_hull has not been called, an empty vector is returned.
   */
  std::vector<Point> get_hull() const { return hull; }

private:
  std::vector<Point> points; ///< Input points for the convex hull algorithm
  std::vector<Point> hull;   ///< Vertices of the computed convex hull
};

#endif
