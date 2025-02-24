#include <iostream>
#include <unordered_map>
#include <string>

int main()
{
	std::unordered_map<std::string, int> unord;
 
	// Insert Few elements in map
	unord.insert( { "First", 1 });
	unord.insert(	{ "Second", 2 });
	unord.insert(	{ "Third", 3 });

	// Overwrite value of an element
	unord["Second"] = 8;
 
	// Iterate Over the unordered_map and display elements
	for (std::pair<std::string, int> element : unord)
		std::cout << element.first << " :: " << element.second << std::endl;

    return 0;
}

