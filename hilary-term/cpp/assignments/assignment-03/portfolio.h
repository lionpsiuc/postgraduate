/**
 * @file portfolio.h
 *
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @author Ion Lipsiuc
 * @date 2025-03-13
 * @version 1.0
 */

#ifndef STATS_H_5IWZAED1
#define STATS_H_5IWZAED1

#include "instruments.h"
#include <vector>

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
double portfolio_profit(std::vector<Trade const *> const &trades,
                        double const S_T);

#endif
