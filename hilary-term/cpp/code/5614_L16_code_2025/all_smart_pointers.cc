#include <iostream>
#include <memory>

int main()
{
    std::weak_ptr<double> wp;

    {
	// Create unique pointer to a double. value=10
	auto up = std::make_unique<double>(10);

	// Construct a shared pointer from that unique_ptr
	std::shared_ptr<double> sp1 {std::move(up)};
	// Create another shared pointer to same double
	auto sp2 = sp1;

	// Set weak_ptr to point to the created double
	wp = sp1;

	// Print how many shared_ptrs to object. i.e ref count
	// This will print "Count: 2"
	std::cout << "Count: " << sp1.use_count() << '\n';

	// Create shared_ptr from the weak_ptr
	auto sp3 = wp.lock();

	// Print how many shared_ptrs to object. i.e ref count
	// This will print "Count: 3"
	std::cout << "Count: " << sp1.use_count() << '\n';
    }

    if(auto sp4 = wp.lock()){
	std::cout << *sp4 << '\n';
    }
    else{
	std::cout << "Dangling pointer\n";
    }

    return 0;
}
