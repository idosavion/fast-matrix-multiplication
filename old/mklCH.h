#ifndef MKL_KS_H
#define MKL_KS_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mkl.h"
#include "ksStrassBasis.h"
#include "utils.h"

double* mkl_multiplication(const double*, const double*, double*, int);
void ksNoRec(double* matrixA, double* matrixB, double* matrixC, double* work);
int compare_ch_and_mkl(int dim);

#endif //MKL_KS_H
