
// Function declarations
// These would usually be in appropriate header file
namespace MyLib1
{
    void library1_function();
    void print_name();
} /* myLib1 */ 

namespace MyLib2
{
    void library2_function();
} /* myLib2 */ 

int main()
{
    MyLib1::library1_function();  // This will be called from lib1
    MyLib2::library2_function();  // This will be called from lib2
    MyLib1::print_name(); 	  // This will be called from lib1
    return 0;
}
