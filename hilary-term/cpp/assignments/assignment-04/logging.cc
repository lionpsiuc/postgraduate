/**
 * @file logging.cc
 * @brief Implementation for logging functions.
 *
 * @author Ion Lipsiuc
 * @version 1.0
 * @date 2025-03-30
 */

#include "logging.h"

#include <sstream>
#include <string>

/**
 * @brief Creates a formatted string with source code location information.
 *
 * @param[in] location The source location, which defaults to the current
 *                     location.
 *
 * @returns A formatted string with file, line, and function information.
 */
std::string sourceline(const std::source_location location) {
  std::ostringstream oss;
  oss << "file: " << location.file_name() << "(" << location.line() << ":"
      << location.column() << ") '" << location.function_name() << "'";
  return oss.str();
}
