/**
 * @file vevox_question.cc
 * @brief This is the code for the question I put up on vevox. Compile and run it yourself to see
 * @author R. Morrin
 * @version 1.0
 * @date 2024-04-05
 */

#include <iostream>

int main()
{
	const int ci {2};
	int *a {const_cast<int *>(&ci)};
	*a = 3;

	std::cout << *a << std::endl;
	
	return 0;
}
