#pragma once
#include <iostream>
#include <cmath>

#include "u_phi_0_transform.hpp"
#include "u_phi_1_transform.hpp"
#include "u_phi_2_transform.hpp"
#include "u_phi_3_transform.hpp"
#include "u_phi_4_transform.hpp"
#include "u_phi_5_transform.hpp"

void transformA(double* original_a, double* transformed_a, int ld, int num_steps, double* work);
size_t get_transformed_A_size(int size, int num_steps);
