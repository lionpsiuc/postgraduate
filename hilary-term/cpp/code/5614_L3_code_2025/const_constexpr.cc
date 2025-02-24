

int main()
{
    //const int x; 			// Error. Must be initialised
    const int x1  	 {1};
    //x1++; 				// Error. Not allowed
    int x2 		 {2};
    const int X3 	 {x2}; 		// OK

    const double d 	 {1.0};
    double e 		 {2.0};
    constexpr double ce1 {3.0}; 	// OK

    constexpr double ce2 {1.0 + 2.0}; 	// OK
    constexpr double ce3 {1.0 + ce1}; 	// OK
    //constexpr double z3  {d + ce1}; 	// ERROR ‘d’ is not usable in a constant expression
    //constexpr double z4 {e + ce1}; 	// ERROR ‘e’ is not usable in a constant expression
    constexpr int z5 	{x1}; 		// OK to initialize constexpr int with const int!!

    // Just put here to eliminate "unused variable warning"
    e +=  x1 + x2 + X3 + d + ce1 + ce2 + ce3 + z5;
    return 0;
}
