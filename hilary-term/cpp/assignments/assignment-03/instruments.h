/**
 * @file instruments.h
 *
 * @brief Header file defining the base Trade class and its derived classes.
 *
 * @author Ion Lipsiuc
 * @date 2025-03-13
 * @version 1.0
 */

#ifndef INSTRUMENTS_H
#define INSTRUMENTS_H
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
   */
  Trade() : cost{0} {
    std::cout << "Trade (base class) Constructor (Default)\n";
  }

  /**
   * @brief Parameterised constructor.
   *
   * @param[in] cost The price or premium paid to enter the trade.
   */
  Trade(double const cost) : cost{cost} {
    std::cout << "Trade (base class) Constructor (overloaded)\n";
  }

  /**
   * @brief Destructor.
   */
  virtual ~Trade() { std::cout << "Trade (base class) Destructor\n"; }

  /**
   * @brief Pure virtual function to calculate the trade's payoff.
   *
   * @param[in] S_T Price of the underlying asset at maturity.
   *
   * @returns The payoff of the trade.
   */
  virtual double payoff(double const S_T) const = 0;

  /**
   * @brief Friend function to calculate total portfolio payoff.
   *
   * @param[in] trades Vector of pointers to Trade objects representing the
   * portfolio.
   * @param[in] S_T Price of the underlying asset at maturity.
   *
   * @returns The total payoff of the portfolio.
   */
  friend double portfolio_payoff(std::vector<Trade const *> const &trades,
                                 double const S_T);

  /**
   * @brief Friend function to calculate total portfolio profit.
   *
   * @param[in] trades Vector of pointers to Trade objects representing the
   * portfolio.
   * @param[in] S_T Price of the underlying asset at maturity.
   *
   * @returns The total profit of the portfolio.
   */
  friend double portfolio_profit(std::vector<Trade const *> const &trades,
                                 double const S_T);

private:
  double const cost; // Holds the premium, or cost paid to enter the trade
};

/**
 * @brief Class representing a forward contract.
 */
class Forward : public Trade {
public:
  // Delete default constructor
  Forward() = delete;

  /**
   * @brief Parameterised constructor.
   *
   * @param[in] fp Price agreed to buy the underlying asset at maturity.
   */
  Forward(double fp) : Trade(), forward_price{fp} {
    std::cout << "Constructor for Forward with forward price " << forward_price
              << "\n";
  }

  /**
   * @brief Destructor.
   */
  ~Forward() override {
    std::cout << "Deleting Forward with forward price " << forward_price
              << "\n";
  }

  /**
   * @brief Calculate the payoff for a forward contract.
   *
   * @param[in] S_T Price of the underlying asset at maturity.
   *
   * @returns The payoff of the forward contract.
   */
  double payoff(double const S_T) const override final {
    return S_T - forward_price;
  }

private:
  double const
      forward_price; // The agreed price to buy the underlying at maturity
};

/**
 * @brief Class representing a call option.
 */
class Call : public Trade {
public:
  // Delete default constructor
  Call() = delete;

  /**
   * @brief Parameterised constructor.
   *
   * @param[in] cost The premium paid to purchase the contract.
   * @param[in] k Strike price of the option.
   */
  Call(double cost, double k) : Trade(cost), strike{k} {
    std::cout << "Creating Call with strike " << strike << ". Premium paid "
              << cost << "\n";
  }

  /**
   * @brief Destructor.
   */
  ~Call() override {
    std::cout << "Destroying Call with strike " << strike << "\n";
  }

  /**
   * @brief Calculate the payoff for a call option.
   *
   * @param[in] S_T Price of the underlying asset at maturity.
   *
   * @returns The payoff of the call option.
   */
  double payoff(double const S_T) const override final {
    return (S_T > strike) ? (S_T - strike) : 0;
  }

private:
  double const strike; // The strike price of the option
};

/**
 * @brief Class representing a put option.
 */
class Put : public Trade {
public:
  // Delete default constructor
  Put() = delete;

  /**
   * @brief Parameterised constructor.
   *
   * @param[in] cost The premium paid to purchase the contract.
   * @param[in] k Strike price of the option.
   */
  Put(double cost, double k) : Trade(cost), strike{k} {
    std::cout << "Creating Put with strike " << strike << ". Premium paid "
              << cost << "\n";
  }

  /**
   * @brief Destructor.
   */
  ~Put() override {
    std::cout << "Destroying Put with strike " << strike << "\n";
  }

  /**
   * @brief Calculate the payoff for a put option.
   *
   * @param[in] S_T Price of the underlying asset at maturity.
   *
   * @returns The payoff of the put option.
   */
  double payoff(double const S_T) const override final {
    return (strike > S_T) ? (strike - S_T) : 0;
  }

private:
  double const strike; // The strike price of the option
};

#endif
