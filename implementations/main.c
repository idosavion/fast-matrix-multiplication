
#include "main.h"

int main(int argc, char** argv)
{
	int i = 16;
	while (i <= 10000)
	{
		printf("Using dim = %d\n", i);
		compare_strassen_and_naive(i);
		mkl_compare(i);
		i = i * 2;
	}
	return 0;

}