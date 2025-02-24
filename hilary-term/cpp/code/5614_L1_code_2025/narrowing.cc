/**
 * @file narrowing.cc
 * @brief Simple ecample showing narrowing
 * @author R. Morrin
 * @version 6.0
 * @date 2025-01-31
 */

int main()
{
    int A {1};
    long B{2};

    int D {B}; 		// Not allowed
    long C {A}; 	// fine

    int E = B; 		// Results in narrowing.

    return 0;
}
