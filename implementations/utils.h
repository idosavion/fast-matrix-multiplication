#ifndef MATMUL_IMPLEMENTATIONS_UTILS_H
#define MATMUL_IMPLEMENTATIONS_UTILS_H

#include <stdio.h>
#include "mkl.h"
//#define INTEL_MKL_THRESH 2048
#define INTEL_MKL_THRESH 2

#define SUBBLOCK_START(mat, i, j, ld, baseDim) &mat[(ld / baseDim * (i - 1) * ld) + (ld/baseDim * (j - 1))]

unsigned memoryRequirements(int blocks, int dim, int base, int numSteps);
void printMatrix(double* matrix, int n);
double* mkl_multiplication(const double*, const double*, double*, int);

#endif // !

