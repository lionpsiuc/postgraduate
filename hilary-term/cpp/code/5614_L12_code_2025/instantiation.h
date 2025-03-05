#ifndef INSTANTIATION_H
#define INSTANTIATION_H


namespace HPC
{


    template <typename T>
    class Instantiation
    {
	private:
	    T X;


	public:
	    Instantiation(T in);
	    ~Instantiation(){};
    };


} /* HPC */ 
#endif /*  */
