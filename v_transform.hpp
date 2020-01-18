#pragma once
#include <iostream>
#include <cmath>

#include "v_psi_0_transform.hpp"
#include "v_psi_1_transform.hpp"
#include "v_psi_2_transform.hpp"
#include "v_psi_3_transform.hpp"
#include "v_psi_4_transform.hpp"
#include "v_psi_5_transform.hpp"

void transformB(double* original_a, double* transformed_a, int ld, int num_steps, double* work);
size_t get_transformed_B_size(int size, int num_steps);
