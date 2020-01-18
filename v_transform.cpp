#include "v_transform.hpp"

static inline int transformed_block_size(int block_size, int from_size, int to_size, int steps_left) {
	int base_block_size = block_size / (pow(from_size, steps_left));
	int power = pow(to_size, steps_left - 1);

	return power * base_block_size;
}



static size_t get_transformed_size(int size, int from_dim, int to_dim, int num_steps) {
	int base_block_size = size / (pow(from_dim, num_steps));
	int power = pow(to_dim, num_steps);

	return power * base_block_size;
}

size_t get_transformed_B_size(int size, int num_steps) {
	return get_transformed_size(size, 9, 20, num_steps);
}

void transformB(double* original_a, double* transformed_a, int size, int num_steps, double* work) {
	int trans_size_2, trans_size_3, trans_size_4, trans_size_5;

	double* res1 = work;
	double* res2 = &res1[size];
	trans_size_2 = get_transformed_size(size, 9, 10, num_steps);
	double* res3 = &res2[trans_size_2];

	trans_size_3 = get_transformed_size(trans_size_2, 10, 11, num_steps);
	double* res4 = &res3[trans_size_3];

	trans_size_4 = get_transformed_size(trans_size_3, 11, 14, num_steps);
	double* res5 = &res4[trans_size_4];

	trans_size_5 = get_transformed_size(trans_size_4, 14, 17, num_steps);
	double* nextwork = &res5[trans_size_5];

	v_psi_0_transform(original_a, res1, sqrt(size), num_steps, nextwork);
	v_psi_1_transform(res1, res2, size, num_steps, nextwork);
	v_psi_2_transform(res2, res3, trans_size_2, num_steps, nextwork);
	v_psi_3_transform(res3, res4, trans_size_3, num_steps, nextwork);
	v_psi_4_transform(res4, res5, trans_size_4, num_steps, nextwork);
	v_psi_5_transform(res5, transformed_a, trans_size_5, num_steps, nextwork);
}
