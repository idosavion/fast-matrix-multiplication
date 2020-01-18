#pragma once
#include <iostream>
#include <cmath>
#include "mkl.h"

void v_psi_0_transform(double* original_a, double* transformed_a, int ld, int steps_left, double* work);
