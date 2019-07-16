#ifndef STRASSEN_WINOGRAD_H
#define STRASSEN_WINOGRAD_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mkl.h"
#include "utils.h"

double* strassensMultiplication(const double* matrixA, const double* matrixB, double* matrixC, int n, int steps_left, double* work);
double* strassensMultRec(const double* matrixA, const double* matrixB, double* matrixC, int n, int steps_left, double* work);
double* create_zero_mat(int dim);
void printMatrix(double*, int);
int compare_strassen_and_mkl(int dim);

#endif //STRASSEN_WINOGRAD_H
