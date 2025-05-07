/**
 * @file  ConvexHull.cc
 * @brief Implementation of ConvexHull class.
 */

#include <algorithm>
#include <vector>

#include "ConvexHull.h"

/**
 * @brief     Constructor that initializes the points from a range defined by
 *            iterators.
 * @param beg Iterator to the beginning of a Point vector.
 * @param end Iterator to the end of a Point vector.
 */
ConvexHull::ConvexHull(std::vector<Point>::iterator beg,
                       std::vector<Point>::iterator end) {

  // Copy the points from the iterator range to our internal vector
  points.assign(beg, end);
}

/**
 * @brief  Generates the convex hull using Andrew's monotone chain algorithm.
 * @return A vector of Points representing the convex hull vertices.
 */
std::vector<Point> ConvexHull::generate_hull() {

  // Return an empty hull if there are fewer than three points
  int n = points.size();
  if (n < 3) {
    hull = points;
    return hull;
  }

  // Temporary vector to store the convex hull vertices
  std::vector<Point> convex_hull;

  // Build lower hull
  for (const auto& point : points) {

    // While we have at least two points in the hull and the last three points
    // make a non-left turn, remove the middle point of the last three
    while (convex_hull.size() >= 2 &&
           !cross_prod(convex_hull[convex_hull.size() - 2], convex_hull.back(),
                       point)) {
      convex_hull.pop_back();
    }
    convex_hull.push_back(point);
  }

  // Build upper hull
  size_t lower_hull_size = convex_hull.size();
  for (auto it = points.rbegin(); it != points.rend(); ++it) {

    // While we have at least two points in the upper hull and the last three
    // points make a non-left turn, remove the middle point of the last three
    while (convex_hull.size() > lower_hull_size &&
           !cross_prod(convex_hull[convex_hull.size() - 2], convex_hull.back(),
                       *it)) {
      convex_hull.pop_back();
    }
    convex_hull.push_back(*it);
  }

  // Remove the last point since it's the same as the first point of the lower
  // hull
  convex_hull.pop_back();

  // Update the hull member variable
  hull = convex_hull;

  return hull;
}

/**
 * @brief       Merges this convex hull with another hull on its right.
 * @param right The ConvexHull to merge with.
 */
void ConvexHull::merge_to_right(ConvexHull& right) {

  // If either hull is empty, return the other
  if (hull.empty()) {
    hull = right.hull;
    return;
  }
  if (right.hull.empty()) {
    return;
  }

  // Find the rightmost point of the left hull
  auto left_rightmost = std::max_element(
      hull.begin(), hull.end(),
      [](const Point& a, const Point& b) { return a.x < b.x; });
  int left_rightmost_idx = std::distance(hull.begin(), left_rightmost);

  // Find the leftmost point of the right hull
  auto right_leftmost = std::min_element(
      right.hull.begin(), right.hull.end(),
      [](const Point& a, const Point& b) { return a.x < b.x; });
  int right_leftmost_idx = std::distance(right.hull.begin(), right_leftmost);

  /*
   * Find the upper tangent
   */

  // Some required variables
  int  upper_left_idx  = left_rightmost_idx;
  int  upper_right_idx = right_leftmost_idx;
  bool upper_found     = false;

  while (!upper_found) {
    upper_found = true;

    // Check if we can move counterclockwise on the left hull
    while (true) {
      int next_idx = (upper_left_idx + 1) % hull.size();
      if (cross_prod(right.hull[upper_right_idx], hull[upper_left_idx],
                     hull[next_idx])) {
        upper_left_idx = next_idx;
        upper_found    = false;
      } else {
        break;
      }
    }

    // Check if we can move clockwise on the right hull
    while (true) {
      int next_idx =
          (upper_right_idx + right.hull.size() - 1) % right.hull.size();
      if (!cross_prod(hull[upper_left_idx], right.hull[upper_right_idx],
                      right.hull[next_idx])) {
        upper_right_idx = next_idx;
        upper_found     = false;
      } else {
        break;
      }
    }
  }

  /*
   * Find the lower tangent
   */

  // Some required variables
  int  lower_left_idx  = left_rightmost_idx;
  int  lower_right_idx = right_leftmost_idx;
  bool lower_found     = false;

  while (!lower_found) {
    lower_found = true;

    // Check if we can move clockwise on the left hull
    while (true) {
      int next_idx = (lower_left_idx + hull.size() - 1) % hull.size();
      if (!cross_prod(right.hull[lower_right_idx], hull[lower_left_idx],
                      hull[next_idx])) {
        lower_left_idx = next_idx;
        lower_found    = false;
      } else {
        break;
      }
    }

    // Check if we can move counterclockwise on the right hull
    while (true) {
      int next_idx = (lower_right_idx + 1) % right.hull.size();
      if (cross_prod(hull[lower_left_idx], right.hull[lower_right_idx],
                     right.hull[next_idx])) {
        lower_right_idx = next_idx;
        lower_found     = false;
      } else {
        break;
      }
    }
  }

  // Merge the hulls using the tangent points
  std::vector<Point> merged_hull;

  // Add points from the left hull and lower tangent to the upper tangent
  size_t idx = lower_left_idx;
  do {
    merged_hull.push_back(hull[idx]);
    idx = (idx + 1) % hull.size();
  } while (idx != (upper_left_idx + 1) % hull.size());

  // Add points from the right hull and the upper tangent to the lower tangent
  idx = upper_right_idx;
  do {
    merged_hull.push_back(right.hull[idx]);
    idx = (idx + 1) % right.hull.size();
  } while (idx != (lower_right_idx + 1) % right.hull.size());

  // Update the hull
  hull = merged_hull;
}
