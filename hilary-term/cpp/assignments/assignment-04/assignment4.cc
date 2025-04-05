/**
 * @file assignment4.cc
 * @brief Main file for the fourth assignment.
 *
 * @author Ion Lipsiuc
 * @version 1.0
 * @date 2025-03-30
 */

#include "matrix-operations.h"
#include "matrix.h"

#include <iostream>

/**
 * @brief Main function.
 *
 * Carries out matrix and vector operations.
 *
 * @returns 0 upon successful execution.
 */
int main() {
  try {
    HPC::Matrix<double> const M1{std::string{"matrix-4x4.txt"}};
    HPC::Matrix<double> const V1{std::string{"vector-4x1.txt"}};
    HPC::Matrix<int> const M2{std::string{"matrix-4x3.txt"}};
    HPC::Matrix<int> const M3{std::string{"matrix-4x4-int.txt"}};
    HPC::Matrix<double> X5{std::string{"matrix-5x5.txt"}};

    // Test 0
    // HPC::Matrix<std::string> V2{std::string * {"vector.txt"}};

    // Test 1
    // HPC::Matrix<double> X1{std::string{"matrix-10x10.txt"}};

    // Test 2
    // HPC::Matrix<double> E{10, 10};
    // E(99, 1) = 1.0;

    // Test 3
    // std::cout << "M1 + M2\n" << M1 + M2 << std::endl;

    // Test 4
    // std::cout << "V1 * M1\n" << V1 * M1 << "\n";

    // Test 5
    // HPC::Matrix<int> I2{M2.get_num_cols(), M2.get_num_rows()};
    // HPC::identity(I2);

    // Test 6
    // HPC::Matrix<int> X2{-10, 10};

    // Test 7
    // HPC::Matrix<int> X3{100'000, 100'000};

    // Test 8
    // HPC::Matrix<double> X4{std::string{"matrix-4x4-corrupt.txt"}};

    // Test 9
    // HPC::Matrix<double> X5{std::string{"matrix-5x5.txt"}};
    // gaussjordan(M1, X);

    // Test 10
    // gaussjordan(M2, V1);

    // Print input data to screen
    std::cout << "\nM1\n" << M1 << "\n";
    std::cout << "\nV1\n" << V1 << "\n";
    std::cout << "\nM2\n" << M2 << "\n";
    std::cout << "\nM3\n" << M3 << "\n";

    // Will call the copy constructor
    auto M1copy{M1};

    // Should call the move constructor
    auto M1move{std::move(M1copy)};

    // Will call operator+
    std::cout << "\nM1 + M3\n" << M1 + M3 << std::endl;

    // Will call operator*
    std::cout << "\nM1 * V1\n" << M1 * V1 << "\n";
    std::cout << "\nM1 * M2\n" << M1 * M2 << "\n";

    // Find inverse of M1
    HPC::Matrix<double> I{M1.get_num_rows(), M1.get_num_cols()};
    HPC::identity(I); // Make the above into an identity matrix
    auto M1inv{gaussjordan(M1, I)};
    std::cout << "\nM1inv\n" << M1inv; // Calculate and print the inverse
    std::cout << "\nCheck M1 * M1inv\n" << M1inv * M1;

    // Solve M1 * x = V1 for x
    auto x{gaussjordan(M1, V1)};
    std::cout << "\nSolve M1 * x = V1 for x\n" << x;
    std::cout << "\nCheck M1 * x = V1\n" << M1 * x;

    // Find the inverse of M3 and solve M3 * x = V1 for x
    HPC::Matrix<int> I3{M3.get_num_rows(), M3.get_num_cols()};
    HPC::identity(I3);
    auto M3inv{gaussjordan(M3, I3)};
    std::cout << "\nM3inv\n" << M3inv;
    auto x2{gaussjordan(M3, V1)};
    std::cout << "\nSolve M3 * x = V1 for x\n" << x2;
    std::cout << "\nCheck M3 * x = V1\n" << M3 * x2;

  } catch (const std::out_of_range &outofrange) {
    std::cerr << "Out of range error: " << outofrange.what() << "\n";
  } catch (const std::string &str) {
    std::cerr << "Exception string caught: " << str << "\n";
  } catch (const char *str) {
    std::cerr << "Exception string caught: " << str << "\n";
  } catch (std::bad_alloc &exception) {
    std::cerr << "bad_alloc detected: " << exception.what();
  } catch (...) {
    std::cerr << "Unknown error caught\n";
  }
  return 0;
}
