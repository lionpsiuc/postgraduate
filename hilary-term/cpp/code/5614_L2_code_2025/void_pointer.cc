

int main()
{
    int a {1}; 		// defines integer "a" and initialise it to hold value 1
    int *b {&a};  	// b now holds the address of a
    void *c {nullptr}; 	// nullptr represents a pointer that does not point to anything
    			// nullptr was introduced in C++11

    //
    //Other code
    //
    
    c = b; 				// This is fine. Can set void* to other pointer.
    //int *d = c; 			// Error. Not allowed. Works fine in C though.

    int *f = (int *)c; 			// This works but is C way of casting. Unsafe.
    int *e = static_cast<int*>(c); 	// This is correct. Will cover this later.

    return 0;
}
