
#include "main.h"

int main(int argc, char** argv)
{
	int i = 32;
	while (i <= 300000)
	{
		printf("Using dim = %d\n", i);
		compare_strassen_and_mkl(i);
		i = i * 2;
	}
	return 0;

}