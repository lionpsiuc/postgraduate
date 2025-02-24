# C++ Programming

## Assignment 2 - Implementing a Gaussian Distribution Class

### Question 1 - Makefile

See `Makefile`.

### Questions 2-6

See `assignment2.cc`.

### Question 7 - AI `class`

The prompt I gave to OpenAI's GP-4o model is as follows:
    
    Can you please generate a C++ class representing a Gaussian distribution? It should be able to print the values of the cumulative distribution function to stdout for given values. Add some examples to the main function so the user can see how it works.

It gave me the following output:

```cpp
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
```

### Question 8 - Written Questions

#### a) Destructor

#### b) Copy vs. Move

#### c) `constexpr` Functions

#### d) Default Parameters
