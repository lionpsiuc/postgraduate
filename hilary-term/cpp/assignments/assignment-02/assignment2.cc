/**
 * @file assignment2.cc
 *
 * @brief Demonstrates the implementation of a Gaussian distribution.
 *
 * This file defines a Gaussian class for representing and manipulating a
 * Gaussian distribution. It includes member functions to compute the
 * probability density function (PDF) and cumulative distribution function
 * (CDF) using various methods (numerical integration, error function, and
 * polynomial approximation). It also demonstrates the use of copy constructors,
 * assignment operators, and free functions.
 *
 * @author Ion Lipsiuc
 * @date 2025-02-19
 * @version 1.0
 */

#include <cmath>
#include <iomanip>
#include <iostream>

const double MYPI{4 * std::atan(1.0)};

/**
 * @class Gaussian
 *
 * @brief A class for representing a Gaussian distribution.
 *
 * The Gaussian class encapsulates the properties of a Gaussian distribution
 * including its mean and standard deviation. It provides methods to compute the
 * PDF and CDF using numerical integration, the complementary error function,
 * and a polynomial approximation. Additionally, it supports copy construction
 * and assignment.
 */
class Gaussian {
public:
  /**
   * @brief Default constructor.
   */
  Gaussian() : mu{0.0}, sigma{1.0} {
    std::cout << "Constructing default with mean 0.0, stdev 1.0" << std::endl;
  }

  /**
   * @brief Constructs a Gaussian distribution with a given mean and standard
   * deviation.
   *
   * @param[in] mean The mean of the distribution.
   * @param[in] stdev The standard deviation of the distribution.
   */
  Gaussian(double const mean, double const stdev) : mu{mean}, sigma{stdev} {
    std::cout << "Constructing with mean " << mean << ", stdev " << stdev
              << std::endl;
  }

  /**
   * @brief Retrieves the mean of the distribution.
   *
   * @returns The mean of the Gaussian distribution.
   */
  double get_mu() const { return mu; }

  /**
   * @brief Retrieves the standard deviation of the distribution.
   *
   * @returns The standard deviation of the Gaussian distribution.
   */
  double get_sigma() const { return sigma; }

  /**
   * @brief Normalises a given value with respect to the distribution.
   *
   * @param[in] x The value to be normalised.
   *
   * @returns The normalised value.
   */
  double normalised(double const x) const { return (x - mu) / sigma; }

  // Prototypes
  Gaussian(const Gaussian &rhs);
  Gaussian &operator=(const Gaussian &rhs);
  double pdf(double const x) const;
  double cdf(double const x, double const left = -8,
             double const step = 1e-3) const;
  double cdf_erfc(double const x) const;
  void print_parameters() const;
  double cdf_poly(double const x) const;

  ~Gaussian() {
    std::cout << "Destroying object with mu = " << mu << " stdev = " << sigma
              << "\n";
  }

private:
  double mu;    // The mean of the distribution
  double sigma; // The standard deviation of the distribution.
};

/**
 * @brief Copy constructor.
 *
 * Creates a new Gaussian object as a copy of an exisiting one.
 *
 * @param[in] rhs The Gaussian object to be copied.
 */
Gaussian::Gaussian(const Gaussian &rhs) : mu(rhs.mu), sigma(rhs.sigma) {
  std::cout << "Copy constructor" << std::endl;
}

/**
 * @brief Copy assignment operator.
 *
 * Assigns the values from the provided Gaussian object to the current
 * object.
 *
 * @param[in] rhs The Gaussian object whose values are to be assigned.
 *
 * @returns A reference to the current object.
 */
Gaussian &Gaussian::operator=(const Gaussian &rhs) {
  if (this != &rhs) {
    mu = rhs.mu;
    sigma = rhs.sigma;
  }
  std::cout << "Copy assignment" << std::endl;
  return *this;
}

// Prototypes
void print_parameters(const Gaussian &dist);
double pdf(const Gaussian &dist, double x);

/**
 * @brief Prints the parameters of the Gaussian distribution.
 */
void Gaussian::print_parameters() const {
  std::cout << "Normal distribution with mean " << mu
            << " and standard deviation " << sigma << std::endl;
}

/**
 * @brief Computes the PDF at a given value.
 *
 * @param[in] x The value at which to evaluate the PDF.
 *
 * @returns The computed PDF value.
 */
double Gaussian::pdf(double const x) const {
  return (1.0 / (sigma * std::sqrt(2 * MYPI))) *
         std::exp(-0.5 * normalised(x) * normalised(x));
}

/**
 * @brief Computes the CDF at a given value using the complementary error
 * function.
 *
 * @param[in] x The value at which to evaluate the CDF.
 *
 * @returns The CDF value.
 */
double Gaussian::cdf_erfc(double const x) const {
  double z = normalised(x);
  return 0.5 * std::erfc(-z / std::sqrt(2.0));
}

/**
 * @brief Compute the CDF at a given value using numerical integration.
 *
 * @param[in] x The value at which to evaluate the CDF.
 * @param[in] left The lower bound for integration.
 * @param[in] step The integration step size.
 *
 * @returns
 */
double Gaussian::cdf(double const x, double const left,
                     double const step) const {
  if (x <= left)
    return 0.0;
  double area = 0.0;
  for (double t = left; t < x; t += step) {
    area += pdf(t) * step;
  }
  return area;
}

/**
 * @brief Computes the CDF using a polynomial approximation.
 *
 * @param[in] x The value at which to evaluate the CDF.
 *
 * @returns The approximate CDF value.
 */
double Gaussian::cdf_poly(const double x) const {
  const double norm{normalised(x)};
  constexpr double boundary{7.07106781186547};
  constexpr double N0{220.206867912376};
  constexpr double N1{221.213596169931};
  constexpr double N2{112.079291497871};
  constexpr double N3{33.912866078383};
  constexpr double N4{6.37396220353165};
  constexpr double N5{0.700383064443688};
  constexpr double N6{3.52624965998911e-02};
  constexpr double M0{440.413735824752};
  constexpr double M1{793.826512519948};
  constexpr double M2{637.333633378831};
  constexpr double M3{296.564248779674};
  constexpr double M4{86.7807322029461};
  constexpr double M5{16.064177579207};
  constexpr double M6{1.75566716318264};
  constexpr double M7{8.83883476483184e-02};
  const double z{std::fabs(norm)};
  double c{0.0};
  if (z <= 37.0) {
    const double e{std::exp(-z * z / 2.0)};
    if (z < boundary) {
      const double n{
          (((((N6 * z + N5) * z + N4) * z + N3) * z + N2) * z + N1) * z + N0};
      const double d{
          ((((((M7 * z + M6) * z + M5) * z + M4) * z + M3) * z + M2) * z + M1) *
              z +
          M0};
      c = e * n / d;
    } else {
      const double f{
          z + 1.0 / (z + 2.0 / (z + 3.0 / (z + 4.0 / (z + 13.0 / 20.0))))};
      c = e / (std::sqrt(2 * MYPI) * f);
    }
  }
  return norm <= 0.0 ? c : 1 - c;
}

/**
 * @brief Prints the parameters of a Gaussian distribution.
 *
 * This free function outputs the mean and standard deviation of the provided
 * Gaussian object.
 *
 * @param[in] dist The Gaussian distribution whose parameters are to be printed.
 */
void print_parameters(const Gaussian &dist) {
  std::cout << "Normal distribution with mean " << dist.get_mu()
            << " and standard deviation " << dist.get_sigma() << std::endl;
}

/**
 * @brief Computes the PDF at a given value.
 *
 * Calls the member function on the Gaussian object.
 *
 * @param[in] dist The Gaussian distribution object.
 * @param[in] x The value at which to compute the PDF.
 *
 * @returns The computed PDF value.
 */
double pdf(const Gaussian &dist, double const x) { return dist.pdf(x); }

/**
 * @brief Main function.
 *
 * Demonstrates the use of the Gaussian class by creating instances, displaying
 * their parameters, and computing the CDF using various methods. It also
 * demonstrates copy construction, assignment, and the use of free functions.
 *
 * @returns 0 upon successful execution.
 */
int main() {
  Gaussian A;
  Gaussian B{1, 2};
  auto list = {-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0};
  std::cout << std::fixed << "\n";
  A.print_parameters();
  std::cout << std::setw(60) << std::setfill('-') << '\n' << std::setfill(' ');
  std::cout << "  x           Phi(x)             Method \n";
  for (auto i : list) {
    std::cout << std::setw(4) << std::setprecision(1) << i << "\t"
              << std::setprecision(14) << A.cdf(i) << "  \tNum. Int.\n"
              << std::setw(4) << std::setprecision(1) << i << "\t"
              << std::setprecision(14) << A.cdf_erfc(i)
              << "  \tUsing std::erfc.\n"
              << std::setw(4) << std::setprecision(1) << i << "\t"
              << std::setprecision(14) << A.cdf_poly(i) << "  \tHart approx.\n";
  }
  std::cout << std::setw(60) << std::setfill('-') << "\n" << std::setfill(' ');
  std::cout << "\n\n";
  B.print_parameters();
  std::cout << std::setw(60) << std::setfill('-') << "\n" << std::setfill(' ');
  for (auto i : list) {
    std::cout << std::setw(4) << std::setprecision(1) << i << "\t"
              << std::setprecision(14) << B.cdf(i, -10.0, 1e-6)
              << "  \tNum. Int.\n"
              << std::setw(4) << std::setprecision(1) << i << "\t"
              << std::setprecision(14) << B.cdf_erfc(i)
              << "  \tUsing std::erfc.\n"
              << std::setw(4) << std::setprecision(1) << i << "\t"
              << std::setprecision(14) << B.cdf_poly(i) << "  \tHart approx.\n";
  }
  std::cout << std::setw(60) << std::setfill('-') << "\n" << std::setfill(' ');
  std::cout << "\n";
  std::cout << "\nUsing free functions\n" << std::endl;
  Gaussian D{2, 5};
  print_parameters(D);
  Gaussian C{D};
  print_parameters(C);
  std::cout << "\n\n";
  A = B;
  print_parameters(A);
  std::cout << std::setprecision(12) << "PDF of A at x=1 is " << pdf(A, 1.0)
            << "\n";
  std::cout << "\n\n" << std::setprecision(1);
  return 0;
}
