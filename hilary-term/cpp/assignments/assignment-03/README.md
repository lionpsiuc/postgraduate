# C++ Programming

## Assignment 3 - Inheritance and Dynamic Polymorphism

### Question 1 - `Makefile`

See `Makefile`.

### Question 2 - Dynamic Polymorphism

See `instruments.h`.

### Question 3 - Non-Member Functions

See `portfolio.h` and `portfolio.cc`.

### Question 4 - Putting It All Together

See `assignment3.cc`.

### Question 5 - Written

#### a) `virtual` and `override`

The `virtual` keyword in the `Trade` `class` means that a function can be overridden by derived classes. We give a simple example below:

```cpp
class Base {
public:
  void print() { std::cout << "Base\n"; }
};

class Derived : public Base {
public:
  void print() { std::cout << "Derived\n"; }
};

int main() {
  Derived test;
  Base *point = &test;
  point->print();
}
```

Compiling and running the above outputs `Base` to the terminal. Let us now use the `virtual` keyword:

```cpp
class Base {
public:
  virtual void print() { std::cout << "Base\n"; }
};

class Derived : public Base {
public:
  void print() { std::cout << "Derived\n"; }
};

int main() {
  Derived test;
  Base *point = &test;
  point->print();
}
```

The above will now print `Derived` to the terminal. `Derived test;` creates an object of type `Derived`. `Base *point = &test;` then creates a pointer of type `Base *` that points to the `Derived` object which is valud due to inheritance. Then, `point->print();` called the `print` function through the pointer, but without `virtual`, the compiler looks at the type of the pointer, and not the actual object type it points to, meaning that at compile-time, the function call is resolved to `Base::print()`.

A disadvantage of using `virtual` is that functions declared as `virtual` require an indirect lookup through the virtual function table during runtime, which can be slightly slower than direct function calls.

Declaring the `payoff` function as pure `virtual` makes the `Trade` `class` an abstract base `class` which cannot be instantiated directly while forcing all derived `class`es to implement the `payoff` function, or else they will also be abstract. It is a way to state that all the derivatives must include a way to calculate their respective payoffs.

The `override` specifier indicates that a function is supposed to `override` a `virtual` function - it provides compile-time checking ensuring that a function signature exactly matches the `virtual` function in a base `class`.

Since it is a compile-time check, there will be no difference in the resulting binary code.

#### b) Polymorphism

#### c) Compiler Issues and Optimisations

##### (i) Defining Own Destructor

##### (ii) Changing to `const`

#### d) Compiler Issues and Optimisations 2

#### e) Miscellaneous

##### (i) Virtual Inheritance

##### (ii) Singleton

### Question 6 - AI Polymorphism
