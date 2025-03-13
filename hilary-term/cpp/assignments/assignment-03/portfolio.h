/**
 * @file portfolio.h
 *
 * @brief Header file for portfolio analysis functions.
 *
 * @author Ion Lipsiuc
 * @date 2025-03-13
 * @version 1.0
 */

#ifndef PORTFOLIO_H
#define PORTFOLIO_H

#include "instruments.h"
#include <vector>

/**
 * @brief Friend function to calculate total portfolio payoff.
 *
 * @param[in] trades Vector of pointers to Trade objects representing the
 * portfolio.
 * @param[in] S_T Price of the underlying asset at maturity.
 *
 * @returns The total payoff of the portfolio.
 */
double portfolio_payoff(std::vector<Trade const *> const &trades,
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
double portfolio_profit(std::vector<Trade const *> const &trades,
                        double const S_T);

#endif
