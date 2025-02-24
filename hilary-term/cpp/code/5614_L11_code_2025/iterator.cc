/**
 * @file iterator.cc
 * @brief  iterator examples for 5614 L9
 * @author R. Morrin
 * @version 2.0
 * @date 2022-03-01
 */
#include <vector>
#include <iostream>


int main()
{
    std::vector <int> X {0, 1, 2, 3, 4};
    const std::vector <double> Y {0.0, 1.1, 2.2, 3.3, 4.4};
    const int max_idx = 4;
    
    int i = 0;

    // regular iterator
    for (std::vector<int>::iterator it = X.begin();  it != X.end(); ++it, ++i) {
       std::cout << "X[" << i << "] = " << *it << '\n';
        (*it)++; 	// Ok
    }
    std::cout << '\n';
    
    i = 0;
    // Constant iterator
    for (std::vector<double>::const_iterator cit = Y.cbegin();  cit != Y.cend(); ++cit, ++i) {
       std::cout << "Y[" << i << "] = " << *cit << '\n';
       // (*it)++; 	// Error. Trying to increment const
    }
    std::cout << '\n';

    i = max_idx;
    // Reverse iterator
    for (std::vector<int>::reverse_iterator rit = X.rbegin();  rit != X.rend(); ++rit, --i) {
       std::cout << "X[" << i << "] = " << *rit << '\n';
    }
    std::cout << '\n';
    
    i = max_idx;
    // Constant reverse iterator
    for (std::vector<double>::const_reverse_iterator crit = Y.crbegin();  crit != Y.crend(); ++crit, --i) {
       std::cout << "Y[" << i << "] = " << *crit << '\n';
    }
    std::cout << '\n';

    i = max_idx;
    // Going backwards using standard iterator. 
    for (std::vector<int>::iterator it1 = X.end();  it1-- != X.begin(); --i) {
       std::cout << "X[" << i << "] = " << *it1 << '\n';
    }
    std::cout << '\n';
    
    i = 0;
    // regular iterator using auto (and also showing std::begin)
    for (auto it = std::begin(X);  it != std::end(X); ++it, ++i) {
       std::cout << "X[" << i << "] = " << *it << '\n';
    }
    std::cout << '\n';
    
    return 0;
}
