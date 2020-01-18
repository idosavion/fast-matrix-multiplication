
            
#pragma once
#include <iostream>
#include <cmath>
#include "mkl.h"

void u_phi_4_transform(double* original_a, double* transformed_a, int size, int steps_left, double* work);

            