#include "utils.h"

unsigned memoryRequirements(int blocks, int dim, int base, int numSteps) {
	unsigned mem = 0;
	int curDenom = base;
	for (int i = 0; i < numSteps; i++) {
		int size = dim / curDenom * dim / curDenom;
		mem += blocks * size;
		curDenom *= base;
	}
	return mem;
}

unsigned CH_memoryRequirements(int blocks, int dim, int base, int numSteps) {
	unsigned mem = 0;
	int curDenom = base;
	for (int i = 0; i < numSteps; i++) {
		int size = dim / curDenom * dim / curDenom;
		mem += blocks * size;
		curDenom *= base;
	}
	return mem;
}

/*
* Function to Print Matrix
*/
void printMatrix(double* matrix, int n)
{
	int i, j;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			printf("   %.4f   ", matrix[i * n + j]);
		}
		printf("\n");
	}
	printf("\n");
}

/*
* Standard Matrix multiplication with O(n^3) time complexity.
*/
double* mkl_multiplication(const double* matrixA, const double* matrixB, double* matrixC, int n)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		n, n, n, 1, matrixA, n, matrixB, n, 0, matrixC, n);
	return matrixC;
}
