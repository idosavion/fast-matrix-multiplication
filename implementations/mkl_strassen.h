#ifndef STRASSEN_H
#define STRASSEN_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mkl.h"
#include "utils.h"

double* strassensMultiplication(const double* matrixA, const double* matrixB, double* matrixC, int n, int steps_left, double* work);
double* strassensMultRec(const double* matrixA, const double* matrixB, double* matrixC, int n, int steps_left, double* work);
// double* divide(double* matrixA, int n, int row, int col);
double* create_zero_mat(int dim);
void printMatrix(double*, int);
// double* addMatrix(double*, double*, int);
// double* subMatrix(double*, double*, int);
// void compose(double*, double*, int, int, int);
int compare_strassen_and_mkl(int dim);

#endif //STRASSEN_H
