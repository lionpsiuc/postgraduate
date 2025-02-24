
// Function declarations
void library1_function();
void library2_function();
void print_name();

int main()
{
    library1_function();  // This will be called from lib1
    library2_function();  // This will be called from lib2
    print_name(); 	  // Ambiguous as to which will be called!!
    return 0;
}
