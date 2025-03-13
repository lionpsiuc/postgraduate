/**
 * @file portfolio.cc
 *
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @author Ion Lipsiuc
 * @date 2025-03-13
 * @version 1.0
 */

#include "portfolio.h"

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
double portfolio_payoff(std::vector<Trade const *> const &trades,
                        double const S_T) {
  double total_payoff{0.0};

  // Iterate through each trade and accumulate the payoffs
  for (auto const &trade : trades) {
    total_payoff += trade->payoff(S_T);
  }

  return total_payoff;
}

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[] trades Explain briefly
 * @param[] S_T Explain briefly.
 *
 * @returns
 */
double portfolio_profit(std::vector<Trade const *> const &trades,
                        double const S_T) {
  double total_payoff = portfolio_payoff(trades, S_T);
  double total_cost{0.0};

  // Calculate the total cost of all trades in the portfolio
  for (auto const &trade : trades) {

    // Access the private cost member through the friend relationship
    total_cost += trade->cost;
  }

  // Profit is payoff minus cost
  return total_payoff - total_cost;
}
