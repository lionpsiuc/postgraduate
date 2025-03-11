#ifndef INSTRUMENTS_H_VK2TICXL
#define INSTRUMENTS_H_VK2TICXL
#include <iostream>
#include <vector>

/**
 * @brief Abstract Base Class Trade
 * 	This class represents the concept of a trade i.e. something we buy, or a
 * contract that we enter into
 */
class Trade {
public:
  // TODO:
  // (1) Define copy and move constructors as default
  // (2) Delete assignment operators

  /**
   * @brief Default construtor for Trade class
   * 	This sets the value of cost equal to zero
   */
  Trade() : cost{0} {
    std::cout << "Trade (base class) Constructor (Default)\n";
  }

  /**
   * @brief Constructor for Trade class
   *
   * @param[in] cost This parameter is the price you paid to purchase, or enter
   * the trade
   */
  Trade(double const cost) : cost{cost} {
    std::cout << "Trade (base class) Constructor (overloaded)\n";
  }

  /**
   * @brief Destructor for Trade
   */
  virtual ~Trade() { std::cout << "Trade (base class) Destructor\n"; }

  // TODO:
  // (3) declare payoff as a pure virtual constant member function
  // (4) Grant access to private member of this class to any non-member
  // functions that need it

private:
  double const cost; ///< Holds the premium, or cost paid to enter the trade
};

/**
 * @brief Class to represent a forward trade
 */

/**
 * @brief Class to represent a Call Option
 */

/**
 * @brief Class to represent a Put Option
 */

#endif /* end of include guard: INSTRUMENTS_H_VK2TICXL */
