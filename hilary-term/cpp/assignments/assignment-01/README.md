# C++ Programming

## Assignment 1 - Refactoring, Memory Management, and Best Practices in C++

### Question 1 - C++ Core Guidelines

#### a) Understanding F.def: Function definitions from the C++ Core Guidelines

- **F.1: 'Package' meaningful operations as carefully named functions**: F.1 in the C++ Core Guidelines is all about writing functions that are small, focused, and have a single clear responsibility. Each function should do one thing and do it well, with its interface (parameters and return value) clearly reflecting its purpose and behavior. This design principle helps make your code easier to understand and maintain because when a function is limited to one task, its internal logic stays simple and its side effects are minimised.
- **F.2: A function should perform a single logical operation**: F.2 emphasises that every function should be focused on a single, logical operation rather than mixing multiple responsibilities. The idea is that when a function is narrowly tailored to one task, it becomes easier to understand, test, and reuse.
- **F.3: Keep functions short and simple**: F.3 emphasises that functions should be kept as short and simple as possible. When a function grows too long or incorporates too many logical paths, it becomes difficult to understand, test, and maintain, increasing the risk of hidden bugs. This minimises the scope of variables and clarifies the code's intent.

> **Summary**: The C++ Core Guidelines emphasise writing functions that are small, focused, and have a clear purpose. F.1 advocates for packaging meaningful operations into well-named functions, ensuring that each function has a distinct responsibility with a clearly defined interface. F.2 reinforces this by stating that a function should perform a single logical operation, avoiding multiple responsibilities that can make it harder to understand, test, and reuse. F.3 further stresses the importance of keeping functions short and simple, as long, complex functions increase the risk of bugs and make maintenance more difficult. By adhering to these principles, code remains readable, maintainable, and less prone to errors.

#### b) Understanding F.call: Parameter passing from the C++ Core Guidelines

- **F.16: For 'in' parameters, pass cheaply-copied types by value and others by reference to const**: F.16 advises that when designing functions, input parameters that are inexpensive to copy should be passed by value, while larger or more complex types should be passed by reference to `const`. This approach not only clearly signals that the function will not modify its input, but also allows both lvalues and rvalues to be used seamlessly. For small, cheap-to-copy types, passing by value avoids the extra level of indirection that comes with references, resulting in simpler and often faster code. Conversely, for types like `std::string` where copying can be expensive, passing by `const` reference is preferred to avoid unnecessary overhead.
- **F.17: For 'in-out' parameters, pass by reference to non-const**: F.17 recommends that functions expecting parameters to be both read and modified should be passed by non-`const` reference. This clearly communicates to the caller that the argument may be changed by the function. Using a non-`const` reference for 'in-out' parameters avoids ambiguity. It is further suggested to send a warning when non-`const` references are not actually modified or are moved, ensuring that the function's behaviour aligns with its signature.
- **F.20: For 'out' output values, prefer return values to output parameters**: F.20 advises that when a function needs to output values, it's generally better to return them rather than using output parameters passed by non-`const` reference. A return value is inherently self-documenting, clearly signalling that it is the result of the function, whereas an output parameter can be ambiguous - it's not always clear if it's intended for input, output, or both.
- **F.21: To return multiple 'out' values, prefer returning a struct**: F.21 recommends that when a function needs to return multiple output values, it's best to encapsulate them in a `struct` rather than using separate output parameters. This approach makes the function's interface self-documenting, clearly indicating which values are being returned, and avoids the ambiguity that can come with using reference parameters for outputs.

> **Summary**: When designing function parameters, the C++ Core Guidelines recommend strategies that enhance clarity, efficiency, and maintainability. F.16 advises passing small, inexpensive-to-copy types by value to avoid unnecessary indirection while passing larger, complex types by `const` reference to prevent expensive copies. F.17 states that parameters expected to be both read and modified should be passed by non-`const` reference, making it explicitly clear to the caller that the function may alter them. F.20 suggests preferring return values over output parameters, as returning a value is more self-explanatory and reduces ambiguity about whether a parameter is for input, output, or both. When multiple values need to be returned, F.21 recommends encapsulating them in a `struct`, ensuring that the function's interface remains clear and self-documenting.

### Question 2 - Reorganising Code

See `reorganise.cc`.

### Question 3 - Using References

See `references.cc`.

### Question 4 - Written

#### a) Using the Code from Question 2 - Reorganising Code

##### (i) Bug in Question 2 - Reorganising Code

The bug is related to file handling. The code you provided to us has a part which checks if the input file for `y` is open when doing a check for the output file stream. It is contained in the following:

```cpp
std::string const y_ofname{"y-even.txt"};
std::ofstream y_outfile{y_ofname};
if (!y_infile.is_open()) {
  std::cerr << "Error opening " << y_ofname << "\n";
  std::exit(EXIT_FAILURE);
}
```

The above checks `y_infile` before attempting to write to `y_outfile`. It should instead be the following:

```cpp
std::string const y_ofname{"y-even.txt"};
std::ofstream y_outfile{y_ofname};
if (!y_outfile.is_open()) {
  std::cerr << "Error opening " << y_ofname << "\n";
  std::exit(EXIT_FAILURE);
}
```

Within our implementation (i.e., `write_array_to_file`), file handling is done correctly and checks the correct file stream (i.e., `outfile`). Moreover, `read_array_from_file` checks `infile`.

The modularisation and separation of different responsibiities within `main` could have helped. The original `main` function was too large. By splitting it into focused functions, as we have done in the modified version, each function handles only one responsibility, making it easier to locate the file handling issue, since each function states exactly what it is supposed to do. Code review becomes easier - debugging something that carries out a singular function is much easier than a monolithic `main` function which does everything. Moreover, as the file handling was being implemented, testing of whether the logic works should have been done - testing to see what happens when non-existent files are attempted to be opened, ensuring that the error handling works as expected. This would have highlighted the bug.

##### (ii) `const` vs. `constexpr` Variables

If `nelem` is declared as `constexpr`, it ensures that `nelem` is always a compile-time constant, opposed to `const`, which indicates that the value of `nelem` cannot change after initialisation. The `const` is useful for variables whose value is known at runtime and does not need to be known at compile-time. Given that we declare its value, `constexpr` is more suitable, allowing us to potentially benefit from compiler optimisations as a result.

#### (b) Using the Code from Question 3 - Using References

Returning a `Vec` for 10,000 elements is far more expensive than returning a `Vec` for 10 elements. For an array of 10 `double`s (assuming a `double` takes eight bytes), the memory taken up is 80 bytes. In contrast, copying 10,000 `double` values takes up 80,000 bytes. We can see that the copy cost increases as the array size increases.

#### (c) Function Overloading

$\alpha$ is not since it only differs by return type. As we saw in the lectures, we can overload on `const` for references and pointer types, but not regular variables. $\beta$ and $\delta$ are allowed since we are changing from a pointer to an `int` (`int *y`) to a pointer to a constant `int` (`int const *y`), and from a reference (`int &z`) to a constant reference (`int const &z`). Moreover, $\gamma$, $\epsilon$, and $\zeta$ are not allowed since it is now a constant pointer to an `int`, it just differs by return type, and `const` cannot be overloaded on a regular variable, respectively.

We can overload on the number of parameters, the types of these parameters, and the sequence of parameter types. In terms of the `const` qualifier, we can overload for references and pointer types, but not regular variables. Moreover, functions that differ only by return type do not consitute overloading.
