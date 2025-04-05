/**
 * @file try-catch.cc
 * @brief Short programme to implement 'try-catch' blocks to process some
 *        exceptions.
 *
 * @author R. Morrin
 * @version 1.0
 * @date 2024-03-09
 */

#include <iostream>
#include <random>

/**
 * @brief Toy function that will throw various types of exceptions.
 *
 * @param[in] in A random number pulled from [0,1).
 */
void throw_fun(const double in) {
  if (in < 0.25) {
    std::cout << "Throwing std::string" << std::endl;
    throw std::string("std::string");
  }
  if (in < 0.5) {
    std::cout << "Throwing C-style string" << std::endl;
    throw "C-style";
  }
  if (in < 0.75) {
    std::cout << "Allocating memory" << std::endl;
    std::size_t s{100'000'000'000};
    [[maybe_unused]] double *t{new double[s]};
  }
  std::cout << "Throwing int" << std::endl;
  throw -1;
}

/**
 * @brief Main function.
 *
 * @returns 0 upon successful execution.
 */
int main() {
  std::random_device rd;
  std::ranlux48 gen{rd()};
  std::uniform_real_distribution<> dist{0.0, 1.0};
  double random_number = dist(gen);
  try {
    throw_fun(random_number);
  } catch (const std::string &str) {
    std::cout << "Exception string caught: " << str << std::endl;
  } catch (const char *str) {
    std::cout << "Exception string caught: " << str << std::endl;
  } catch (const std::exception &e) {
    std::cout << "bad_alloc detected: " << e.what() << std::endl;
  } catch (int value) {
    std::cout << "Exception int caught: " << value << std::endl;
  } catch (...) {
    std::cout << "Unknown error caught" << std::endl;
  }

  return 0;
}
