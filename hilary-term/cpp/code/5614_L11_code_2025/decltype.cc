#include <iostream>
template <typename T>
class Vector
{
public:
    Vector (int n) : sz{n} , data{new T [sz]}{} ;
    int get_size() const { return sz;};
    T get_by_idx(int i) const { return data[i];};
    void set_by_idx(int i, T in){ data[i] = in;};

private:
    int sz;
    T * data;
};

template<typename T, typename U>
Vector<decltype(T{}+U{})> operator+(const Vector<T>& a, const Vector<U>& b){
    std::cout << "Adding\n";
    if(a.get_size() != b.get_size()){
	std::cerr << "Cannot add vectors of different length\n";
	exit(-1);
    }
    int N = a.get_size();
    Vector<decltype(T{}+U{})> res {N};
    for (int i=0; i< a.get_size(); ++i) {
	res.set_by_idx(i, a.get_by_idx(i) + b.get_by_idx(i));
    }
    return res;
}

int main()
{
    Vector<int> A (3);
    Vector<double> B (3); 

    for (auto i = 0; i < 3; ++i) {
	A.set_by_idx(i,i);
	B.set_by_idx(i,i+0.1);
    }

    auto res = A+B;
    //decltype(A+B) res = A+B;   // This would also work here

    for (auto i=0; i< res.get_size(); ++i) {
	std::cout << "res[" << i <<"] = " << res.get_by_idx(i) << '\n';
    }
 
    return 0;
}
