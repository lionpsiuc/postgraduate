/**
 * @file assignment3.cc
 *
 * @brief The main programme showcasing the use of inheritance and dynamic
 * polymorhpism.
 *
 * @author Ion Lipsiuc
 * @date 2025-03-13
 * @version 1.0
 */

#include "instruments.h"
#include "portfolio.h"
#include <iostream>
#include <vector>

/**
 * @brief Main function.
 *
 * Creates a portfolio of trades and prints the payoff and profit for each.
 *
 * @returns 0 upon successful execution.
 */
int main() {
  std::vector<Trade const *> trades;

  // Add some trades
  trades.push_back(new const Forward{6.0});
  trades.push_back(new const Forward{2.0});
  trades.push_back(new const Call{1.0, 6.0});
  trades.push_back(new const Call{5.5, 3.0});
  trades.push_back(new const Put{4.0, 7.0});
  trades.push_back(new const Put{4.5, 6.0});

  std::cout << "\nPortfolio:\nS_T" << "\t" << "Payoff" << "\t" << "Profit\n";
  for (int stock_price = 0; stock_price <= 15; ++stock_price) {
    std::cout << stock_price << "\t" << portfolio_payoff(trades, stock_price)
              << "\t" << portfolio_profit(trades, stock_price) << "\n";
  }
  std::cout << "\n";

  // Delete allocated trades manually
  for (auto i = 0u; i < trades.size(); ++i) {
    delete trades[i];
  }

  return 0;
}
