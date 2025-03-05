#include <memory>

int main()
{
    // Using new to create smart pointer
    std::unique_ptr<double> upn {new double};
    std::shared_ptr<double> spn {new double};

    // Better way. Use make functions. This can
    // also save some typing if you use auto.
    auto upm {std::make_unique<double>()};
    auto spm {std::make_shared<double>()};
    return 0;
}
