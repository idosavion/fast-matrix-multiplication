#ifndef STRASSEN_H
#define STRASSEN_H
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"

#include <time.h>

double* strassensMultiplication(const double*, const double*, double* , int, int);
double* mkl_multiplication(const double*, const double*, double*, int);
double* strassensMultRec(const double*, const double*, double*, int n, int);
// double* divide(double* matrixA, int n, int row, int col);
double* create_zero_mat(int dim);
void printMatrix(double*, int);
// double* addMatrix(double*, double*, int);
// double* subMatrix(double*, double*, int);
// void compose(double*, double*, int, int, int);
int compare_strassen_and_mkl(int dim);

#endif //STRASSEN_H
