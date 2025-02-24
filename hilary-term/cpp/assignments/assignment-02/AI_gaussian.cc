#include <cmath>
#include <iomanip>
#include <iostream>

class Gaussian {
private:
  double mu;
  double sigma;

public:
  Gaussian(double mean = 0.0, double std_dev = 1.0)
      : mu(mean), sigma(std_dev) {}
  double cdf(double x) const {
    return 0.5 * (1 + std::erf((x - mu) / (sigma * std::sqrt(2.0))));
  }
};

int main() {
  Gaussian g1;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Standard Normal Distribution (mean=0, std_dev=1):\n";
  std::cout << "CDF(-1) = " << g1.cdf(-1) << "\n";
  std::cout << "CDF(0)  = " << g1.cdf(0) << "\n";
  std::cout << "CDF(1)  = " << g1.cdf(1) << "\n\n";
  Gaussian g2(1.0, 2.0);
  std::cout << "Gaussian Distribution (mean=1, std_dev=2):\n";
  std::cout << "CDF(-1) = " << g2.cdf(-1) << "\n";
  std::cout << "CDF(1)  = " << g2.cdf(1) << "\n";
  std::cout << "CDF(3)  = " << g2.cdf(3) << "\n";
  return 0;
}