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

The `override` specifier (used in derived `class`es) indicates that a function is supposed to use a `virtual` function - it provides compile-time checking ensuring that a function signature exactly matches the `virtual` function in a base `class`.

Since it is a compile-time check, there will be no difference in the resulting binary code.

#### b) Polymorphism

Static polymorphism, also known as compile-time polymorphism, is where the compler determines which function or operation to call based on the number, types, and order of arguments. It is concerned with overloaded functons. Dynamic polymorphism, also known as runtime polymorphism, is concerned with `virtual` functions seen in `class`es. Here, the function called is determined at runtime based on the object type instead of the reference or pointer type. Moreover, dynamic polymorphism involves inheritance whereas static polymorphism does not.

Within our assignment, we created an abstract `class` called `Trade` from which all derived classes (i.e., `Forward`, `Call`, and `Put`) are instantiated. It declares a pure `virtual` function `payoff` (i.e., `virtual double payoff(double const S_T) const = 0;`) stating that all derived `class`es must implement a method to calculate their respective payoffs. Each derived `class` then implements its own version of the `payoff` function. Moreover, within the `portfolio_payoff` function, the line `total_payoff += trade->payoff(S_T);` shows dynamic polymorphism since the actual object type (i.e., `Forward`, `Call`, or `Put`) is determined at runtime (where we define `std::vector<Trade const *> trades;` within our `main` function and use the `push_back` function to store our base `class` pointers).

#### c) Compiler Issues and Optimisations

##### (i) Defining Own Destructor

When defining our own destructor, the compiler no longer declares the move assignment and move operator, as per the matrix seen in the lectures. This is evident in the fact that `Widget(Widget &&)` and `operator=(Widget &&)` are no longer seen in the filtered symbol table.

In terms of efficiency, operations which would have been cheap pointer swaps (thanks to the move semantics) become expensive deep copies. This is more significant in our case, since our `struct` contains a list (i.e., `std::list<int> vals;`) which means that copy operations require allocating new memory for all the elements.

Line 22 still works since the compiler the compiler first looks for a move constructor (i.e., `Widget(Widget &&)`), but since we defined a destructor, it cannot find a move constructor, so it falls back to using a copy constructor and makes a complete duplicate of `X`, including its list. For line 24, the compiler tries to find a move assignment (i.e., `Widget &operator=(Widget &&)`), but it cannot and thus, it falls back to using a copy assignment, making a complete duplicate of `Z`, requiring memory allocation and copying.

##### (ii) Changing to `const`

The code will not compile. We are making `original_N` a `const` member variable, meaning that we cannot modify it after initialisation. In the line `Z = Y;` and `X = std::move(Z);`, we are attempting to change the members of the object to match another object (i.e., an assignment operator, regardless if it a move or copy, modifies the existing object). We would have to provide custom assignment operators in this case.

#### d) Compiler Issues and Optimisations 2

For the first change, we don't see `Move Constructor` printed to the terminal when we simply have `return (incremented);` since the compiler carries out return value optimisation (RVO) for us (i.e., we are forcing the compiler to use the move constructor when calling `return std::move(incremented);`, preventing it from using RVO). Since we are forcing a move constructor call, that is why we see `Move Constructor` at the end of the second output. Indeed, the warning given (`warning: moving a local object in a return statement prevents copy elision`) is directly telling us that we are making performance worse and that we are preventing copy elision (i.e., RVO).

Now, when commenting out the move constructor, we are seeing an example of the compiler falling back to use the copy constructor when the move constructor is not defined (recall that if the copy constructor is defined by the user, the move constructor is not declared, hence why the compiler doesn't create a default one despite us commenting it out). When we reach `return (std::move(incremented));`, our compiler tries to look for a move constructor to handle the created rvalue reference, doesn't, so it falls back to using the copy constructor instead. This is why we see `Copy Constructor`. Note that if we use `return (incremented);` instead, we would not see `Copy Constructor` since the compiler will use RVO.

From what I have seen in this question, using `std::move` to return function values is not necessary as it prevents compiler optimisations from occurring.

#### e) Miscellaneous

##### (i) Virtual Inheritance

`virtual` inheritance is needed when we have a so-called 'diamond' inheritance pattern where a `class` inherits from two `class`es that both inherit from a common base `class`. Without `virtual` inheritance, the derived class would contain two separate copies of the base `class`. For example, if `B` and `C` are derived `class`es from `A`, and `D` is derived from both `B` and `C`, then, without `virtual` inheritance, `D` would contain two copies of `A`. In terms of the code, we have:

```cpp
class A { /* */ };
class B : virtual public A { /* */ };
class C : virtual public A { /* */ };
class D : public B, public C { /* */ };
```

##### (ii) Singleton

A singleton pattern is a design pattern that restricts the instantiation of a `class` to exactly one object. We would define a `private` constructor, ensuring to prevent direct instantiation of the class from outside (i.e., `private: Singleton() { /* */ };`), delete the copy and move operations which will prevent copying or moving the single instance (i.e., `Singleton(Singleton const &) = delete;` and `Singleton &operator=(Singleton const &) = delete;`), create a `static` method to access the instance returning a reference to prevent copying (i.e., `static Singleton &getInstance();`), and use a `static` local variable for storage (i.e., `static Singleton instance;`).

The `static` keyword makes the `getInstance` function callable without an existing object and makes the `instance` variable persist across function calls.

### Question 6 - AI Polymorphism

Using ChatGPT o3-mini-high, I asked it the following:

    Can you please generate an example of polymorphism in C++, creating base and derived classes. Make sure it is thorough.

The answer it gave me can be seen in `ai-polymorphism.cc`.

In terms of a quick and simple explanation, we can see that it generated a base `Animal` `class`. It defines a `virtual` function `makeSound` which means that derived classes can provide their own implementation of it. A `virtual` destructor is also used to ensure that when a derived object is deleted through a pointer to `Animal`, the proper destructor is called. We can see two derived `class`es, `Dog` and `Cat`. Both these `class`es inherit from `Animal` and `override` the `makeSound` method using `override`. When `makeSound` is called on an object of the derived `class`es via a pointer to the base `class`, the appropriate derived `class`'s method is used. We can see some polymorphic behaviour in our `main` function - we create a vector of `Animal *` pointers, iterate over it and calling `makeSound` on each pointer will use dynamic binding whereby the call resolves to the correct overridden function based on the object's actual type.

We notice that the `class`es do not have explicitly deifned constructors meaning that the compiler will provide a default one which simply initialises the object. Our base `class` explicitly defines a `virtual` destructor, ensuring that when an object of a derived `class` is deleted via a pointer to `Animal`, the destructor of the derived class is invoked and while the derived `class`es do not have explicitly defined destructors, default ones are generated by the compiler. Moreover, since we define a destructor for our base class, this means that move operators will not be declared by the compiler thus implying that any move operations will default to copy operations.

Our inheritance is `public` meaning that `public` members of our base class remain `public` in `Dog` and `Cat`. Our other options (i.e., `protected`, `private`, and `virtual`) are not applicable here.

One thing I would change about this example is to change our base `class` to an abstract one, since having just an `Animal` class is not very interesting and should serve as the building blocks for other, more specific, `class`es. This can be done by declaring at least one of its member functions as a pure `virtual` function (in this case, we only have one member function). This can be done by assigning `= 0` at the end of the declaration. We would make the following change:

```cpp
class Animal {
public:
  // Pure virtual function to make Animal abstract
  virtual void makeSound() const = 0;
}
```

Note that now we cannot do `animals.push_back(new Animal());` in our `main` function since `Animal` is now abstract and cannot be instantiated. We do not need `override` in the derived `class`es, but it is useful to ensure compile-time checking.

One might suggest using the rule of three since we have explicitly defined a destructor, however, it is empty and does not perform any special tasks. It is used in a polymorphic context (i.e., ensuring we use the correct destructor) so I do not believe it is necessary to follow this rule. Moreover, the rule of zero is followed in the derived `class`es. Having said all of this, you did suggest we use the rule of five defaults in our assignments, which this does not follow.
