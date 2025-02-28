#include <cmath>
#include <iostream>
#include <random>
#include <vector>

class GaussianDistribution {
private:
  double mean;
  double stdDev;
  std::mt19937 generator;
  std::normal_distribution<double> distribution;

public:
  // Constructor with default parameters
  GaussianDistribution(double mean = 0.0, double stdDev = 1.0)
      : mean(mean), stdDev(stdDev) {

    // Initialise random number generator with random device
    std::random_device rd;
    generator = std::mt19937(rd());

    // Create normal distribution with given parameters
    distribution = std::normal_distribution<double>(mean, stdDev);
  }

  // Generate a single random sample from the distribution
  double sample() { return distribution(generator); }

  // Generate multiple samples
  std::vector<double> sampleVector(int n) {
    std::vector<double> samples(n);
    for (int i = 0; i < n; i++) {
      samples[i] = sample();
    }
    return samples;
  }

  // Calculate probability density function (PDF) at x
  double pdf(double x) const {
    double exponent = -0.5 * std::pow((x - mean) / stdDev, 2);
    double coefficient = 1.0 / (stdDev * std::sqrt(2.0 * M_PI));
    return coefficient * std::exp(exponent);
  }

  // Calculate cumulative distribution function (CDF) at x
  double cdf(double x) const {
    return 0.5 * (1.0 + std::erf((x - mean) / (stdDev * std::sqrt(2.0))));
  }

  // Calculate the quantile function (inverse CDF) for a given probability
  double quantile(double p) const {
    if (p <= 0.0 || p >= 1.0) {
      throw std::invalid_argument("Probability must be between 0 and 1");
    }

    // This is an approximation of the inverse error function
    double t = std::sqrt(-2.0 * std::log(std::min(p, 1.0 - p)));

    double c0 = 2.515517;
    double c1 = 0.802853;
    double c2 = 0.010328;
    double d1 = 1.432788;
    double d2 = 0.189269;
    double d3 = 0.001308;
    double sign = (p < 0.5) ? -1.0 : 1.0;
    double numerator = c0 + c1 * t + c2 * t * t;
    double denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    return mean + stdDev * sign * (t - numerator / denominator);
  }

  // Getters and setters
  double getMean() const { return mean; }
  double getStdDev() const { return stdDev; }
  void setMean(double newMean) {
    mean = newMean;
    distribution = std::normal_distribution<double>(mean, stdDev);
  }
  void setStdDev(double newStdDev) {
    if (newStdDev <= 0) {
      throw std::invalid_argument("Standard deviation must be positive");
    }

    stdDev = newStdDev;
    distribution = std::normal_distribution<double>(mean, stdDev);
  }
};

int main() {

  // Create a normal distribution with mean 5 and std dev 2
  GaussianDistribution gauss(5.0, 2.0);

  // Generate a random sample
  double sample = gauss.sample();
  std::cout << "Random sample: " << sample << std::endl;

  // Calculate PDF at x = 6
  double pdf_at_6 = gauss.pdf(6.0);
  std::cout << "PDF at x=6: " << pdf_at_6 << std::endl;

  // Calculate CDF at x = 6
  double cdf_at_6 = gauss.cdf(6.0);
  std::cout << "CDF at x=6: " << cdf_at_6 << std::endl;

  return 0;
}
