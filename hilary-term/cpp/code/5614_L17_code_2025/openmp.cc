#include <functional>
#include <thread>
#include <array>
#include <iostream>
#include <omp.h>

void f(){
  std::cout << "Function f()" << std::endl;
}

struct Hello
{
  void operator()(){
    std::cout << "Function object" << std::endl;
  }
};

int main()
{
  auto named_lambda = [](){
    std::cout << "Named Lambda"  << std::endl;
  };

  std::array<std::function<void()>, 4> fns{f
    , Hello{}
    , named_lambda
    , [](){std::cout << "Anon lambda" << std::endl;}};
  /*
   *fns[0] = f;
   *fns[1] = Hello{};
   *fns[2] = named_lambda;
   *fns[3] = [](){std::cout << "Anon lambda" << std::endl;};
   */

#pragma omp parallel num_threads(4)
  {
    fns[omp_get_thread_num()]();
  }

  return 0;
}
