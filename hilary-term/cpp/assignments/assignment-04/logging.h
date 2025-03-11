/**
 * @file logging.h
 * @brief Header file for logging functions.
 * 		See https://en.cppreference.com/w/cpp/utility/source_location
 * @author R. Morrin
 * @version 3.0
 * @date 2025-02-25
 */
#ifndef LOGGING_H_QYJGNTWO
#define LOGGING_H_QYJGNTWO
#include <source_location>

std::string sourceline(
    const std::source_location location = std::source_location::current());

#endif /* end of include guard: LOGGING_H_QYJGNTWO */
