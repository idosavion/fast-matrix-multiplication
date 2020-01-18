#pragma once
//#include <mkl_cblas.h>
//#include <mkl.h>


void recursive_bilinear_algorithm(double* a, double* b, int steps_left, int size_of_block, double* result);
int run_recursive_bilinear_algorithm();

void print_vector(double* vector, int len);
