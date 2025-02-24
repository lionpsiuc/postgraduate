/**
 * @file reorganise.cc
 *
 * @brief Following guidelines given in C++ Core Guidelines.
 *
 * We organise the original programme to ensure it follows the C++ Core
 * Guidelines, specifically F.def: Function definitions with a specific focus on
 * F.1 through to F.3, inclusive. This programme reads arrays from files,
 * processes them to filter positive numbers and even numbers, and writes the
 * filtered results to new files.
 *
 * @author Ion Lipsiuc
 * @date 2025-02-11
 * @version 1.0
 */

#include <fstream>
#include <iostream>
#include <string>

/**
 * @brief Filters even numbers from an array.
 *
 * @param[in] input The input array of integers.
 * @param[in] n The size of the input array.
 * @param[out] output The output array where even numbers will be stored.
 *
 * @returns Number of even numbers in the array.
 */
std::size_t filter_even_numbers(int const input[], std::size_t const n,
                                int output[]) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (input[i] % 2 == 0) { // Check if the number is even
      output[count++] = input[i];
    }
  }
  return count;
}

/**
 * @brief Filters positive numbers from an array.
 *
 * @param[in] input The input array of doubles.
 * @param[in] n The size of the input array.
 * @param[out] output The output array where positive numbers will be stored.
 *
 * @returns Number of positive numbers in the array.
 */
std::size_t filter_positive_numbers(double const input[], std::size_t const n,
                                    double output[]) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (input[i] > 0) { // Only store positive numbers
      output[count++] = input[i];
    }
  }
  return count;
}

/**
 * @brief Prints an array of doubles in a formatted manner.
 *
 * @param[in] v The array of doubles.
 * @param[in] n The size of the array.
 */
void pretty_print_vec(double const v[], std::size_t const n) {
  if (n > 0) {
    for (std::size_t i = 0; i < n - 1; ++i) {
      std::cout << v[i] << ", ";
    }
    std::cout << v[n - 1];
  }
}

/**
 * @brief Prints an array of integers in a formatted manner.
 *
 * @param[in] v The array of integers.
 * @param[in] n The size of the array.
 */
void pretty_print_vec(int const v[], std::size_t const n) {
  if (n > 0) {
    for (std::size_t i = 0; i < n - 1; ++i) {
      std::cout << v[i] << ", ";
    }
    std::cout << v[n - 1];
  }
}

/**
 * @brief Reads an array of doubles from a file.
 *
 * @param[in] filename The name of the file to read from.
 * @param[out] v The array where the read values will be stored.
 * @param[in] n The number of elements to read.
 */
void read_array_from_file(std::string const &filename, double v[],
                          std::size_t const n) {
  std::ifstream infile{filename}; // Open file for reading
  if (!infile.is_open()) {        // Check if file opening failed
    std::cerr << "Error opening " << filename << '\n';
    std::exit(EXIT_FAILURE);
  }
  for (std::size_t i = 0; i < n; ++i) {
    infile >> v[i]; // Read values into array
  }
  infile.close(); // Close the file
}

/**
 * @brief Reads an array of integers from a file.
 *
 * @param[in] filename The name of the file to read from.
 * @param[out] v The array where the read values will be stored.
 * @param[in] n The number of elements to read.
 */
void read_array_from_file(std::string const &filename, int v[],
                          std::size_t const n) {
  std::ifstream infile{filename}; // Open file for reading
  if (!infile.is_open()) {        // Check if file opening failed
    std::cerr << "Error opening " << filename << '\n';
    std::exit(EXIT_FAILURE);
  }
  for (std::size_t i = 0; i < n; ++i) {
    infile >> v[i]; // Read values into array
  }
  infile.close(); // Close the file
}

/**
 * @brief Writes an array of doubles to a file.
 *
 * @param[in] filename The name of the file to write to.
 * @param[in] v The array of doubles to be written.
 * @param[in] n The number of elements in the array.
 */
void write_array_to_file(std::string const &filename, double const v[],
                         std::size_t const n) {
  std::ofstream outfile{filename}; // Open file for writing
  if (!outfile.is_open()) {        // Check if file opening failed
    std::cerr << "Error opening " << filename << '\n';
    std::exit(EXIT_FAILURE);
  }
  for (std::size_t i = 0; i < n; ++i) {
    outfile << v[i] << '\n'; // Write each value to the file
  }
  outfile.close(); // Close the file
}

/**
 * @brief Writes an array of integers to a file.
 *
 * @param[in] filename The name of the file to write to.
 * @param[in] v The array of integers to be written.
 * @param[in] n The number of elements in the array.
 */
void write_array_to_file(std::string const &filename, int const v[],
                         std::size_t const n) {
  std::ofstream outfile{filename}; // Open file for writing
  if (!outfile.is_open()) {        // Check if file opening failed
    std::cerr << "Error opening " << filename << '\n';
    std::exit(EXIT_FAILURE);
  }
  for (std::size_t i = 0; i < n; ++i) {
    outfile << v[i] << '\n'; // Write each value to the file
  }
  outfile.close(); // Close the file
}

/**
 * @brief Main function.
 *
 * Carries out the above functions.
 *
 * @returns 0 upon successful execution.
 */
int main(void) {
  constexpr std::size_t nelem{10};
  double x[nelem];
  read_array_from_file("x.txt", x, nelem);
  std::cout << "Original x vector: {";
  pretty_print_vec(x, nelem);
  std::cout << "}\n";
  double x_pos[nelem];
  std::size_t num_pos = filter_positive_numbers(x, nelem, x_pos);
  write_array_to_file("x-positive.txt", x_pos, num_pos);
  int y[nelem];
  read_array_from_file("y.txt", y, nelem);
  std::cout << "Original y vector: {";
  pretty_print_vec(y, nelem);
  std::cout << "}\n";
  int y_even[nelem];
  std::size_t num_even = filter_even_numbers(y, nelem, y_even);
  write_array_to_file("y-even.txt", y_even, num_even);
  return 0;
}
