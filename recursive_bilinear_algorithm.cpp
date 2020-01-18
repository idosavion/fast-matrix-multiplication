#include "recursive_bilinear_algorithm.h"
#include <iostream>
#include <cmath>
#include "mkl.h"
#include "mkl_measure.cpp"
#include "bs_recursive.hpp"
#include "u_transform.hpp"
#include "v_transform.hpp"
#include "w_transform.hpp"
#include <iomanip>

void print_vector(double* vector, int len)
{
    std::cout << std::fixed;
    std::cout << std::setprecision(1);
    
	for (int i = 0; i < len; i++)
	{
		std::cout <<  vector[i] << ", ";
	}
	std::cout << std::endl;
}

static inline int transformed_block_size(int block_size, int steps_left) {
	int base_block_size = block_size / (pow(9, steps_left));
	int pow_of_ten = pow(10, steps_left);

	return pow_of_ten * base_block_size;
}

int run_recursive_bilinear_algorithm() {
	int dim = 9;
	double* work = (double*)mkl_malloc(sizeof(double) * 20 * 2000, 64);//just randomly assigned a size that seemed big enough, you may need to tweak it
	int num_steps = 1;

	double* original_a = (double*)mkl_malloc(sizeof(double) * dim*dim, 64);
	double* transformed_a = (double*)MKL_malloc(sizeof(double) * get_transformed_A_size(dim * dim, 2), 64);
	double* original_b = (double*)mkl_malloc(sizeof(double) * dim * dim, 64);
	double* transformed_b = (double*)MKL_malloc(sizeof(double) * get_transformed_B_size(dim * dim, 2), 64);
	double* transformed_c = (double*)MKL_malloc(sizeof(double) * get_transformed_C_size(dim * dim, 2), 64);
	double* output_c = (double*)mkl_malloc(sizeof(double) * dim * dim, 64);
	

	for (int i = 0; i < dim*dim; ++i) {
		original_a[i] = 1;
		original_b[i] = 1;
		output_c[i] = 0;
	}
	
	int transformed_size = get_transformed_A_size(dim * dim, num_steps);
	for (int i = 0; i < get_transformed_C_size(dim * dim, 2); i++) {
		transformed_c[i] = 0;
	}

	transformA(original_a, transformed_a, dim* dim, num_steps, work);
	transformB(original_b, transformed_b, dim* dim, num_steps, work);
	recursive_decomposed_bs(transformed_a, transformed_b, transformed_c, transformed_size, num_steps, work);
	transformC(transformed_c, output_c, transformed_size, num_steps, work);


	for (size_t i = 0; i < dim*dim; i++)
	{
		std::cout << output_c[i] << " ";
	}
	std::cout << std::endl;
	return 0;
}
