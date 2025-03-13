/**
 * @file instruments.h
 *
 * @brief Header file defining the base Trade class and its derived classes.
 *
 * @author Ion Lipsiuc
 * @date 2025-03-13
 * @version 1.0
 */

#ifndef INSTRUMENTS_H_VK2TICXL
#define INSTRUMENTS_H_VK2TICXL
#include <iostream>
#include <vector>

/**
 * @brief Base class representing a generic financial trade.
 *
 * Serves as the foundation for the derived classes. It manages the cost of
 * entering the trade.
 */
class Trade {
public:
  // Define copy and move constructors as default
  Trade(Trade const &) = default;
  Trade(Trade &&) = default;

  // Delete assignment operators
  Trade &operator=(Trade const &) = delete;
  Trade &operator=(Trade &&) = delete;

  /**
   * @brief Default constructor.
   *
   * Initialises a trade with zero cost and prints a message stating that the
   * constructor was called.
   */
  Trade() : cost{0} {
    std::cout << "Trade (base class) Constructor (Default)\n";
  }

  /**
   * @brief Parameterised constructor.
   *
   * Initialises a trade with the given cost and prints a message stating that
   * the constructor was called.
   *
   * @param[in] cost The price or premium paid to enter the trade.
   */
  Trade(double const cost) : cost{cost} {
    std::cout << "Trade (base class) Constructor (overloaded)\n";
  }

  /**
   * @brief Descructor.
   *
   * Ensures that the derived class destructors are called correctly and outputs
   * a message indicating that the destructor was called.
   */
  virtual ~Trade() { std::cout << "Trade (base class) Destructor\n"; }

  /**
   * @brief Pure virtual function to calculate the trade's payoff.
   *
   * Further explanation, if required.
   *
   * @param[] S_T Explain briefly.
   */
  virtual double payoff(double const S_T) const = 0;

  /**
   * @brief Explain briefly.
   *
   * Further explanation, if required.
   *
   * @param[] trades Explain briefly.
   * @param[] S_T Explain briefly.
   *
   * @returns
   */
  friend double portfolio_payoff(std::vector<Trade const *> const &trades,
                                 double const S_T);

  /**
   * @brief Explain briefly.
   *
   * Further explanation, if required.
   *
   * @param[] trades Explain briefly.
   * @param[] S_T Explain briefly.
   *
   * @returns
   */
  friend double portfolio_profit(std::vector<Trade const *> const &trades,
                                 double const S_T);

private:
  double const cost; // Holds the premium, or cost paid to enter the trade
};

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 */
class Forward : public Trade {
public:
  // Delete default constructor
  Forward() = delete;

  /**
   * @brief Explain briefly.
   *
   * Further explanation, if required.
   *
   * @param[] fp Explain briefly.
   */
  Forward(double fp) : Trade(), forward_price{fp} {
    std::cout << "Constructor for Forward with forward price " << forward_price
              << "\n";
  }

  /**
   * @brief Explain briefly.
   *
   * Further explanation, if required.
   */
  ~Forward() override {
    std::cout << "Deleting Forward with forward price " << forward_price
              << "\n";
  }

  /**
   * @brief Explain briefly.
   *
   * Further explanation, if required.
   *
   * @param[] S_T Explain briefly.
   *
   * @returns
   */
  double payoff(double const S_T) const override final {
    return S_T - forward_price;
  }

private:
  double const
      forward_price; // The agreed price to buy the underlying at maturity
};

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 */
class Call : public Trade {
public:
  // Delete default constructor
  Call() = delete;

  /**
   * @brief Explain briefly.
   *
   * Further explanation, if required.
   *
   * @param[] cost Explain briefly.
   * @param[] k Explain briefly.
   */
  Call(double cost, double k) : Trade(cost), strike{k} {
    std::cout << "Creating Call with strike " << strike << ". Premium paid "
              << cost << "\n";
  }

  /**
   * @brief Explain briefly.
   *
   * Further explanation, if required.
   */
  ~Call() override {
    std::cout << "Destroying Call with strike " << strike << "\n";
  }

  /**
   * @brief Explain briefly.
   *
   * Further explanation, if required.
   *
   * @param[] S_T Explain briefly.
   *
   * @returns
   */
  double payoff(double const S_T) const override final {
    return (S_T > strike) ? (S_T - strike) : 0;
  }

private:
  double const strike; // The strike price of the option
};

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 */
class Put : public Trade {
public:
  // Delete default constructor
  Put() = delete;

  /**
   * @brief Explain briefly.
   *
   * Further explanation, if required.
   *
   * @param[] cost Explain briefly.
   * @param[] k Explain briefly.
   */
  Put(double cost, double k) : Trade(cost), strike{k} {
    std::cout << "Creating Put with strike " << strike << ". Premium paid "
              << cost << "\n";
  }

  /**
   * @brief Explain briefly.
   *
   * Further explanation, if required.
   */
  ~Put() override {
    std::cout << "Destroying Put with strike " << strike << "\n";
  }

  /**
   * @brief Explain briefly.
   *
   * Further explanation, if required.
   *
   * @param[] S_T Explain briefly.
   *
   * @returns
   */
  double payoff(double const S_T) const override final {
    return (strike > S_T) ? (strike - S_T) : 0;
  }

private:
  double const strike; // The strike price of the option
};

#endif
