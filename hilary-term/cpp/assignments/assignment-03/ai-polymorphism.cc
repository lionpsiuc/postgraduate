#include <iostream>
#include <vector>

class Animal {
public:
  // Virtual function to allow derived classes to override it
  virtual void makeSound() const {
    std::cout << "Animal makes a generic sound." << std::endl;
  }

  // A virtual destructor ensures that the destructors of derived classes
  // are called correctly when an object is deleted via a base class pointer
  virtual ~Animal() {}
};

class Dog : public Animal {
public:
  // Override the makeSound function for Dog
  void makeSound() const override { std::cout << "Woof! Woof!" << std::endl; }
};

class Cat : public Animal {
public:
  // Override the makeSound function for Cat
  void makeSound() const override { std::cout << "Meow! Meow!" << std::endl; }
};

int main() {

  // Creating a vector of pointers to Animal objects
  std::vector<Animal *> animals;

  // Adding different animal objects to the vector
  animals.push_back(new Animal());
  animals.push_back(new Dog());
  animals.push_back(new Cat());

  // Iterate over the vector and call makeSound on each object.
  for (const Animal *animal : animals) {
    animal->makeSound();
  }

  // Delete allocated memory to avoid memory leaks
  for (Animal *animal : animals) {
    delete animal;
  }

  return 0;
}
