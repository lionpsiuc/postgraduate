/**
 * @file logging.h
 * @brief Header file for logging functions.
 *
 * @author Ion Lipsiuc
 * @version 1.0
 * @date 2025-03-30
 */

#ifndef LOGGING_H
#define LOGGING_H

#include <source_location>
#include <string>

/**
 * @brief Creates a formatted string with source code location information.
 *
 * @param[in] location The source location, which defaults to the current
 *                     location.
 *
 * @returns A formatted string with file, line, and function information.
 */
std::string sourceline(
    const std::source_location location = std::source_location::current());

#endif
