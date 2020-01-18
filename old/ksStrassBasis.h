#ifndef KS_STRASS_BASIS_H
#define KS_STRASS_BASIS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mkl.h"
#include "utils.h"

int benchmarkKSBasisTransform(int dim);
void ksBasisTransformRec(double* matrixA, int n, int steps_left, double* work);
void ksInverseBasisTransformRec(double* matrixA, int n, int steps_left, double* work);

#endif //STRASSEN_H