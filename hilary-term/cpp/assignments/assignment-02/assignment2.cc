/**
 * @file assignment2.cc
 *
 * @brief Explain briefly.
 *
 * Further explanation, if required.
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
 * @class
 *
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 */
class Gaussian {
public:
  /**
   * @brief Explain briefly.
   *
   * Further explanation, if required.
   */
  Gaussian() : mu{0.0}, sigma{1.0} {
    std::cout << "Constructing default with mean 0.0, stdev 1.0" << std::endl;
  }

  /**
   * @brief Explain briefly.
   *
   * Further explanation, if required.
   *
   * @param[in] mean Explain briefly.
   * @param[in] stdev Explain briefly.
   */
  Gaussian(double const mean, double const stdev) : mu{mean}, sigma{stdev} {
    std::cout << "Constructing with mean " << mean << ", stdev " << stdev
              << std::endl;
  }

  /**
   * @brief Explain briefly.
   *
   * Further explanation, if required.
   *
   * @returns
   */
  double get_mu() const { return mu; }

  /**
   * @brief Explain briefly.
   *
   * Further explanation, if required.
   *
   * @returns
   */
  double get_sigma() const { return sigma; }

  /**
   * @brief Explain briefly.
   *
   * Further explanation, if required.
   *
   * @param[in] x Explain briefly.
   *
   * @returns
   */
  double normalised(double const x) const { return (x - mu) / sigma; }

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
  double mu;
  double sigma;
};

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in] rhs Explain briefly.
 */
Gaussian::Gaussian(const Gaussian &rhs) : mu(rhs.mu), sigma(rhs.sigma) {
  std::cout << "Copy constructor" << std::endl;
}

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in] rhs Explain briefly.
 *
 * @returns
 */
Gaussian &Gaussian::operator=(const Gaussian &rhs) {
  if (this != &rhs) {
    mu = rhs.mu;
    sigma = rhs.sigma;
  }
  std::cout << "Copy assignment" << std::endl;
  return *this;
}

void print_parameters(const Gaussian &dist);
double pdf(const Gaussian &dist, double x);

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 */
void Gaussian::print_parameters() const {
  std::cout << "Normal distribution with mean " << mu
            << " and standard deviation " << sigma << std::endl;
}

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in] x Explain briefly.
 *
 * @returns
 */
double Gaussian::pdf(double const x) const {
  return (1.0 / (sigma * std::sqrt(2 * MYPI))) *
         std::exp(-0.5 * normalised(x) * normalised(x));
}

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in] x Explain briefly.
 *
 * @returns
 */
double Gaussian::cdf_erfc(double const x) const {
  double z = normalised(x);
  return 0.5 * std::erfc(-z / std::sqrt(2.0));
}

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in] x Explain briefly.
 * @param[in] left Explain briefly.
 * @param[in] step Explain briefly.
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
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in] x Explain briefly.
 *
 * @returns
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
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in] dist Explain briefly.
 */
void print_parameters(const Gaussian &dist) {
  std::cout << "Normal distribution with mean " << dist.get_mu()
            << " and standard deviation " << dist.get_sigma() << std::endl;
}

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in] dist Explain briefly.
 * @param[in] x Explain briefly.
 *
 * @returns
 */
double pdf(const Gaussian &dist, double const x) { return dist.pdf(x); }

/**
 * @brief Main function.
 *
 * Further explanation, if required.
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
