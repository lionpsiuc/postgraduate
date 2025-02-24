/**
 * @file 5614_L7_question.cc
 * @brief Put on slide as a question for the class
 * @author R. Morrin
 * @version 1.0
 * @date 2025-02-12
 */
#include <iostream>

class Widget
{
    int a;
    friend void printWidget(Widget& in);

    public:
    Widget() : a{100} {};
};

struct Other {
    int a;
    friend void printOther(Widget& in);

    Other() : a{2} {};
};

void printWidget(Widget& in){
    std::cout << in.a;
}

void printWidget2(Widget& in){
    std::cout << in.a;
}

void printOther(Other& in){
    std::cout << in.a;
}

void printOther2(Other& in){
    std::cout << in.a;
}



int main()
{
    Widget A{};
    Other B{};
    printWidget(A);
    //printWidget2(A);
    printOther(B);
    printOther2(B);


    return 0;
}












