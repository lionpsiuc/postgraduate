#include <iostream>
#include <unistd.h>
#include <sys/wait.h>


void f(){
    for(auto i=0; i<3; ++i){
	std::cout << "Function f(). Process ID = " << getpid() << '\t' << i <<  std::endl;
	sleep (1);
    }
}

void g(){
    for(auto i=0; i<3; ++i){
	std::cout << "Function g(). Process ID = " << getpid() << '\t' << i <<  std::endl;
	sleep (1);
    }
}


int main(void)
{
    pid_t ch;

    std::cout << "Process ID = " << getpid() << std::endl;

    if((ch = fork()) < 0){
	std::cerr << "Error forking\n";
    }
    else if(ch==0){  // Child process
	f();
    }
    else{
	pid_t ch2;

	if((ch2 = fork()) < 0){  // 2nd child
	    std::cerr << "Error forking\n";
	}
	else if(ch2==0){
	    g();
	}
    }

    int status;
    wait(&status);
    wait(&status);
    return 0;
}
