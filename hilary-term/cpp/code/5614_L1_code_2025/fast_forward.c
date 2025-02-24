#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Struct to hold a student's data
 */
struct Student {
	char name[64]; 		/*< Student's name 			*/
	int num_courses; 	/*< Number of courses student is taking */
	double *grades; 	/*< Array of grades received 		*/
};


/**
 * @brief  Function to create a struct Student
 * 	   The function allocates memory for the structure itself, and the array of grades
 *
 * @param[in] ng  number of grades to allocate space for
 *
 * @return  pointer to the allocated struct
 */
struct Student* create_Student(int const nc){
	struct Student *s1 = malloc(sizeof *s1);
	if(s1==NULL){
		perror("Error allocating memory for s1");
		exit(EXIT_FAILURE);
	}
	s1->grades = malloc(nc * sizeof *s1->grades);
	if(s1->grades==NULL){
		perror("Error allocating memory for s1");
		exit(EXIT_FAILURE);
	}
	s1->num_courses = nc;
	return s1;
}

/**
 * @brief Function to free memory allocated to a struct Student
 *
 * @param in Pointer to the struct Student you want to free.
 */
void free_Student(struct Student *in){
	free(in->grades);
	free(in);
}


/**
 * @brief Function to deep copy a struct Students
 *
 * @param in Pointer to the original structure
 *
 * @return  Pointer to the created deep copy
 */
struct Student * deep_copy_Student(struct Student *in){
	struct Student *newstr = create_Student(in->num_courses);

	for (int i = 0; i < in->num_courses; i++) {
		newstr->grades[i] = in->grades[i];	
	}
	strcpy(newstr->name, in->name);
	return newstr;
}


int main(void)
{
	struct Student *s1 = malloc(sizeof *s1);
	if(s1==NULL){
		perror("Error allocating memory for s1");
		exit(EXIT_FAILURE);
	}
	s1->grades = malloc(2 * sizeof *s1->grades);
	if(s1->grades==NULL){
		perror("Error allocating memory for s1");
		exit(EXIT_FAILURE);
	}
	sprintf(s1->name, "Joe Bloggs");
	s1->num_courses = 2;
	s1->grades[0] = 0.75;
	s1->grades[0] = 0.82;

	/*
	 * Do work
	 */

	struct Student *s2 = create_Student(2);
	s2->grades[0] = 0.55;
	s2->grades[1] = 0.65;

	struct Student *s3 = deep_copy_Student(s1);

	free_Student(s2);
	free_Student(s3);


	free(s1->grades);
	free(s1);


	return 0;
}
