#include "w_transform.hpp"

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

size_t get_transformed_C_size(int size, int num_steps) {
	return get_transformed_size(size, 9, 20, num_steps);
}

void transformC(double* original_c, double* output_c, int size, int num_steps, double* work) {
	int trans_size_1, trans_size_2, trans_size_3, trans_size_4, trans_size_5;

	double* res5 = work;
	trans_size_5 = get_transformed_size(size, 20, 17, num_steps);
	
	double* res4 = &res5[trans_size_5];
	trans_size_4 = get_transformed_size(trans_size_5, 17, 14, num_steps);

	double* res3 = &res4[trans_size_4];
	trans_size_3 = get_transformed_size(trans_size_4, 14, 11, num_steps);
	
	double* res2 = &res3[trans_size_3];
	trans_size_2 = get_transformed_size(trans_size_3, 11, 10, num_steps);
	
	double* res1 = &res2[trans_size_2];
	trans_size_1 = get_transformed_size(trans_size_2, 10, 9, num_steps);

	double* nextwork = &res1[trans_size_1];

	w_tau_5_transform(original_c, res5, size, num_steps, nextwork);
	w_tau_4_transform(res5, res4, trans_size_5, num_steps, nextwork);
	w_tau_3_transform(res4, res3, trans_size_4, num_steps, nextwork);
	w_tau_2_transform(res3, res2, trans_size_3, num_steps, nextwork);
	w_tau_1_transform(res2, res1, trans_size_2, num_steps, nextwork);
	w_tau_0_transform(res1, output_c, trans_size_1, num_steps, nextwork);
}
