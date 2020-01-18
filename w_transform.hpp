#pragma once
#include <iostream>
#include <cmath>

#include "w_tau_0_transform.hpp"
#include "w_tau_1_transform.hpp"
#include "w_tau_2_transform.hpp"
#include "w_tau_3_transform.hpp"
#include "w_tau_4_transform.hpp"
#include "w_tau_5_transform.hpp"

void transformC(double* original_c, double* output_c, int size, int num_steps, double* work);
size_t get_transformed_C_size(int size, int num_steps);
