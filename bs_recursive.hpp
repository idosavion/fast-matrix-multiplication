#pragma once
#include <iostream>
#include <cmath>
#include "mkl.h"

void recursive_decomposed_bs(double* matrixA, double* matrixB, double* matrixC, int size, int steps_left, double* work);
