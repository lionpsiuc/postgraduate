/**
 * @file  ConvexHull.cc
 * @brief Implementation of ConvexHull class.
 */

#include <algorithm>
#include <limits>
#include <vector>

#include "ConvexHull.h"

/**
 * Constructor that initializes the points from a range defined by iterators.
 *
 * @param beg Iterator to the beginning of a Point vector.
 * @param end Iterator to the end of a Point vector.
 */
ConvexHull::ConvexHull(std::vector<Point>::iterator beg,
                       std::vector<Point>::iterator end) {

  // Copy the points from the iterator range to our internal vector
  points.assign(beg, end);
}

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
std::vector<Point> ConvexHull::generate_hull() {
  int n = points.size();
  if (n < 3) {

    // Return points as is for fewer than three points
    hull = points;
    return hull;
  }

  // Temporary vector to store the convex hull vertices
  std::vector<Point> temp_hull;

  // Build lower hull
  for (const auto& pt : points) {

    // Remove points that make a right turn or are collinear
    while (temp_hull.size() >= 2 &&
           cross_prod(temp_hull[temp_hull.size() - 2], temp_hull.back(), pt)) {
      temp_hull.pop_back();
    }

    temp_hull.push_back(pt);
  }

  // Build upper hull
  size_t t = temp_hull.size();

  // Process points in reverse order for upper hull
  for (int i = n - 2; i >= 0; --i) {
    const auto& pt = points[i];

    // Remove points that make a right turn or are collinear
    while (temp_hull.size() >= t + 1 &&
           cross_prod(temp_hull[temp_hull.size() - 2], temp_hull.back(), pt)) {
      temp_hull.pop_back();
    }

    temp_hull.push_back(pt);
  }

  // Remove the duplicate starting point if hull has more than one point
  if (temp_hull.size() > 1) {
    temp_hull.pop_back();
  }

  hull = temp_hull;
  return hull;
}

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
void ConvexHull::merge_to_right(ConvexHull& right) {
  // Handle empty hulls
  if (this->hull.empty()) {
    this->hull = right.hull;
    // Steal points if right hull is not empty
    if (!right.hull.empty()) {
      this->points.insert(this->points.end(),
                          std::make_move_iterator(right.points.begin()),
                          std::make_move_iterator(right.points.end()));
      right.points.clear();
      right.hull.clear();
    }
    return;
  }
  if (right.hull.empty()) {
    // Nothing to merge if right hull is empty
    return;
  }

  // Find rightmost point on left hull and leftmost point on right hull
  auto left_rightmost = std::max_element(
      this->hull.begin(), this->hull.end(),
      [](const Point& a, const Point& b) { return a.x < b.x; });
  int   left_rightmost_idx = std::distance(this->hull.begin(), left_rightmost);
  Point A                  = this->hull[left_rightmost_idx];

  auto right_leftmost = std::min_element(
      right.hull.begin(), right.hull.end(),
      [](const Point& a, const Point& b) { return a.x < b.x; });
  int   right_leftmost_idx = std::distance(right.hull.begin(), right_leftmost);
  Point B                  = right.hull[right_leftmost_idx];

  // Define vertical line between hulls
  double x_mid = (A.x + B.x) / 2.0;

  // Calculate initial line parameters
  double m, c, y_mid;
  if (B.x - A.x == 0) {
    // Handle vertical line case
    m     = std::numeric_limits<double>::infinity();
    c     = A.x;
    y_mid = (A.y + B.y) / 2.0;
  } else {
    m     = (B.y - A.y) / (B.x - A.x);
    c     = A.y - m * A.x;
    y_mid = m * x_mid + c;
  }

  // Find upper tangent
  int  upper_left_idx      = left_rightmost_idx;
  int  upper_right_idx     = right_leftmost_idx;
  bool found_upper_tangent = false;

  while (!found_upper_tangent) {
    found_upper_tangent         = true;
    int current_upper_right_idx = upper_right_idx;

    // Check all points on right hull for better upper tangent
    for (size_t i = 0; i < right.hull.size(); ++i) {
      int next_right_idx_candidate =
          (current_upper_right_idx + i) % right.hull.size();
      if (next_right_idx_candidate == upper_right_idx && i > 0)
        continue;

      Point  next_right = right.hull[next_right_idx_candidate];
      double next_y_mid_val;

      // Calculate intersection y value
      if (next_right.x - this->hull[upper_left_idx].x == 0) {
        next_y_mid_val = (this->hull[upper_left_idx].y + next_right.y) / 2.0;
      } else {
        double next_m = (next_right.y - this->hull[upper_left_idx].y) /
                        (next_right.x - this->hull[upper_left_idx].x);
        double next_c = this->hull[upper_left_idx].y -
                        next_m * this->hull[upper_left_idx].x;
        next_y_mid_val = next_m * x_mid + next_c;
      }

      // For upper tangent, maximize y_mid
      if (next_y_mid_val > y_mid) {
        upper_right_idx     = next_right_idx_candidate;
        y_mid               = next_y_mid_val;
        found_upper_tangent = false;
      }
    }

    // Check left hull for better upper tangent
    Point C_candidate     = right.hull[upper_right_idx];
    bool  changed_on_left = false;
    for (size_t i = 0; i < this->hull.size(); ++i) {
      int next_left_idx_candidate =
          (upper_left_idx + this->hull.size() - i) % this->hull.size();
      if (next_left_idx_candidate == upper_left_idx && i > 0)
        continue;

      Point  next_left = this->hull[next_left_idx_candidate];
      double next_y_mid_val;

      // Calculate intersection y value
      if (C_candidate.x - next_left.x == 0) {
        next_y_mid_val = (next_left.y + C_candidate.y) / 2.0;
      } else {
        double next_m =
            (C_candidate.y - next_left.y) / (C_candidate.x - next_left.x);
        double next_c  = next_left.y - next_m * next_left.x;
        next_y_mid_val = next_m * x_mid + next_c;
      }

      // For upper tangent, maximize y_mid
      if (next_y_mid_val > y_mid) {
        upper_left_idx      = next_left_idx_candidate;
        y_mid               = next_y_mid_val;
        found_upper_tangent = false;
        changed_on_left     = true;
      }
    }
    if (changed_on_left)
      found_upper_tangent = false;
  }

  // Find lower tangent (similar to upper but we minimize y_mid)
  int  lower_left_idx      = left_rightmost_idx;
  int  lower_right_idx     = right_leftmost_idx;
  bool found_lower_tangent = false;

  // Reset initial intersection for lower tangent
  if (B.x - A.x == 0) {
    y_mid = (A.y + B.y) / 2.0;
  } else {
    m     = (B.y - A.y) / (B.x - A.x);
    c     = A.y - m * A.x;
    y_mid = m * x_mid + c;
  }

  while (!found_lower_tangent) {
    found_lower_tangent        = true;
    int current_lower_left_idx = lower_left_idx;

    // Check left hull for better lower tangent
    for (size_t i = 0; i < this->hull.size(); ++i) {
      int next_left_idx_candidate =
          (current_lower_left_idx + i) % this->hull.size();
      if (next_left_idx_candidate == lower_left_idx && i > 0)
        continue;

      Point  next_left = this->hull[next_left_idx_candidate];
      double next_y_mid_val;

      // Calculate intersection y value
      if (right.hull[lower_right_idx].x - next_left.x == 0) {
        next_y_mid_val = (next_left.y + right.hull[lower_right_idx].y) / 2.0;
      } else {
        double next_m = (right.hull[lower_right_idx].y - next_left.y) /
                        (right.hull[lower_right_idx].x - next_left.x);
        double next_c  = next_left.y - next_m * next_left.x;
        next_y_mid_val = next_m * x_mid + next_c;
      }

      // For lower tangent, minimize y_mid
      if (next_y_mid_val < y_mid) {
        lower_left_idx      = next_left_idx_candidate;
        y_mid               = next_y_mid_val;
        found_lower_tangent = false;
      }
    }

    // Check right hull for better lower tangent
    Point D_candidate      = this->hull[lower_left_idx];
    bool  changed_on_right = false;
    for (size_t i = 0; i < right.hull.size(); ++i) {
      int next_right_idx_candidate =
          (lower_right_idx + right.hull.size() - i) % right.hull.size();
      if (next_right_idx_candidate == lower_right_idx && i > 0)
        continue;

      Point  next_right = right.hull[next_right_idx_candidate];
      double next_y_mid_val;

      // Calculate intersection y value
      if (next_right.x - D_candidate.x == 0) {
        next_y_mid_val = (D_candidate.y + next_right.y) / 2.0;
      } else {
        double next_m =
            (next_right.y - D_candidate.y) / (next_right.x - D_candidate.x);
        double next_c  = D_candidate.y - next_m * D_candidate.x;
        next_y_mid_val = next_m * x_mid + next_c;
      }

      // For lower tangent, minimize y_mid
      if (next_y_mid_val < y_mid) {
        lower_right_idx     = next_right_idx_candidate;
        y_mid               = next_y_mid_val;
        found_lower_tangent = false;
        changed_on_right    = true;
      }
    }
    if (changed_on_right)
      found_lower_tangent = false;
  }

  // Create the merged hull by walking around both hulls
  std::vector<Point> merged_hull;

  // Add points from left hull: upper tangent to lower tangent (clockwise)
  int current_idx = upper_left_idx;
  while (true) {
    merged_hull.push_back(this->hull[current_idx]);
    if (current_idx == lower_left_idx)
      break;
    current_idx = (current_idx + 1) % this->hull.size();
  }

  // Add points from right hull: lower tangent to upper tangent (clockwise)
  current_idx = lower_right_idx;
  while (true) {
    merged_hull.push_back(right.hull[current_idx]);
    if (current_idx == upper_right_idx)
      break;
    current_idx = (current_idx + 1) % right.hull.size();
  }

  // Update hull and steal points from right
  this->hull = merged_hull;
  this->points.insert(this->points.end(),
                      std::make_move_iterator(right.points.begin()),
                      std::make_move_iterator(right.points.end()));

  // Clear right hull and points
  right.points.clear();
  right.hull.clear();
}
