/**
 * @file hpc-concepts.h
 * @brief Header file defining the Number concept for use in template
 *        constraints.
 *
 * @author Ion Lipsiuc
 * @version 1.0
 * @date 2025-03-30
 */

#ifndef HPC_CONCEPTS_H
#define HPC_CONCEPTS_H

/**
 * @brief Concept that constrains types to those supporting arithmetic
 *        operations.
 *
 * @tparam T The type to check against the Number concept.
 */
template <typename T>
concept Number = requires(T a, T b) {
  a + b;
  a - b;
  a *b;
  a / b;
  a += b;
  a -= b;
  a *= b;
  a /= b;
};

#endif
