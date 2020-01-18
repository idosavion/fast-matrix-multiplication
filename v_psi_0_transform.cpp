#include "v_psi_0_transform.hpp"

const int FROM_DIM = 9;
const int TO_DIM = 9;


static inline int transformed_block_size(int block_size, int steps_left) {
	int base_block_size = block_size / (pow(FROM_DIM, steps_left));
	int power = pow(TO_DIM, steps_left - 1);

	return power * base_block_size;
}


static void S1_sum(const double* matrixA, double* S1, int ld) {
	int block_width = ld / 3;
	int block_height = block_width;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < block_height; j++) {
		for (int k = 0; k < block_width; k++) {
			S1[j * (block_width)+k] = matrixA[((1 * block_height) + j) * ld + ((1 * block_width) + k)] - matrixA[((2 * block_height) + j) * ld + ((1 * block_width) + k)];
		}
	}
}

static void S2_sum(const double* matrixA, double* S2, int ld) {
	int block_width = ld / 3;
	int block_height = block_width;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < block_height; j++) {
		for (int k = 0; k < block_width; k++) {
			S2[j * (block_width)+k] = matrixA[j * ld + ((2 * block_width) + k)] + matrixA[((1 * block_height) + j) * ld + ((2 * block_width) + k)] - matrixA[((2 * block_height) + j) * ld + ((2 * block_width) + k)];
		}
	}
}

static void S3_sum(const double* matrixA, double* S3, int ld) {
	int block_width = ld / 3;
	int block_height = block_width;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < block_height; j++) {
		for (int k = 0; k < block_width; k++) {
			S3[j * (block_width)+k] = matrixA[((1 * block_height) + j) * ld + ((2 * block_width) + k)] - matrixA[((2 * block_height) + j) * ld + k] - matrixA[((2 * block_height) + j) * ld + ((1 * block_width) + k)];
		}
	}
}

static void S4_sum(const double* matrixA, double* S4, int ld) {
	int block_width = ld / 3;
	int block_height = block_width;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < block_height; j++) {
		for (int k = 0; k < block_width; k++) {
			S4[j * (block_width)+k] = -matrixA[j * ld + k] - matrixA[((1 * block_height) + j) * ld + k];
		}
	}
}

static void S5_sum(const double* matrixA, double* S5, int ld) {
	int block_width = ld / 3;
	int block_height = block_width;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < block_height; j++) {
		for (int k = 0; k < block_width; k++) {
			S5[j * (block_width)+k] = matrixA[((2 * block_height) + j) * ld + k] + matrixA[((2 * block_height) + j) * ld + ((1 * block_width) + k)] - matrixA[((2 * block_height) + j) * ld + ((2 * block_width) + k)];
		}
	}
}

static void S6_sum(const double* matrixA, double* S6, int ld) {
	int block_width = ld / 3;
	int block_height = block_width;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < block_height; j++) {
		for (int k = 0; k < block_width; k++) {
			S6[j * (block_width)+k] = matrixA[j * ld + k] - matrixA[((1 * block_height) + j) * ld + ((1 * block_width) + k)] + matrixA[((2 * block_height) + j) * ld + ((1 * block_width) + k)];
		}
	}
}

static void S7_sum(const double* matrixA, double* S7, int ld) {
	int block_width = ld / 3;
	int block_height = block_width;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < block_height; j++) {
		for (int k = 0; k < block_width; k++) {
			S7[j * (block_width)+k] = -matrixA[((2 * block_height) + j) * ld + ((1 * block_width) + k)] + matrixA[((2 * block_height) + j) * ld + ((2 * block_width) + k)];
		}
	}
}

static void S8_sum(const double* matrixA, double* S8, int ld) {
	int block_width = ld / 3;
	int block_height = block_width;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < block_height; j++) {
		for (int k = 0; k < block_width; k++) {
			S8[j * (block_width)+k] = -matrixA[j * ld + k] - matrixA[j * ld + ((1 * block_width) + k)] - matrixA[((1 * block_height) + j) * ld + ((2 * block_width) + k)];
		}
	}
}

static void S9_sum(const double* matrixA, double* S9, int ld) {
	int block_width = ld / 3;
	int block_height = block_width;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < block_height; j++) {
		for (int k = 0; k < block_width; k++) {
			S9[j * (block_width)+k] = -matrixA[j * ld + ((1 * block_width) + k)] + matrixA[j * ld + ((2 * block_width) + k)] - matrixA[((1 * block_height) + j) * ld + ((1 * block_width) + k)] + matrixA[((1 * block_height) + j) * ld + ((2 * block_width) + k)] + matrixA[((2 * block_height) + j) * ld + ((1 * block_width) + k)] - matrixA[((2 * block_height) + j) * ld + ((2 * block_width) + k)];
		}
	}
}
void v_psi_0_transform(double* original_a, double* transformed_a, int ld, int steps_left, double* work) {
	int block_ld = ld / 3;
	int block_size = block_ld * block_ld;

	if (steps_left == 0) {
		return;
	}
	else if (steps_left == 1) {
		S1_sum(original_a, transformed_a, ld);
		S2_sum(original_a, &transformed_a[1 * block_size], ld);
		S3_sum(original_a, &transformed_a[2 * block_size], ld);
		S4_sum(original_a, &transformed_a[3 * block_size], ld);
		S5_sum(original_a, &transformed_a[4 * block_size], ld);
		S6_sum(original_a, &transformed_a[5 * block_size], ld);
		S7_sum(original_a, &transformed_a[6 * block_size], ld);
		S8_sum(original_a, &transformed_a[7 * block_size], ld);
		S9_sum(original_a, &transformed_a[8 * block_size], ld);
		return;
	}

	double* S = work;
	double* nextWork = &work[block_size];

	S1_sum(original_a, S, ld);
	v_psi_0_transform(S, transformed_a, block_ld, steps_left - 1, nextWork);

	S2_sum(original_a, S, ld);
	v_psi_0_transform(S, &transformed_a[1 * block_size], block_ld, steps_left - 1, nextWork);

	S3_sum(original_a, S, ld);
	v_psi_0_transform(S, &transformed_a[2 * block_size], block_ld, steps_left - 1, nextWork);

	S4_sum(original_a, S, ld);
	v_psi_0_transform(S, &transformed_a[3 * block_size], block_ld, steps_left - 1, nextWork);

	S5_sum(original_a, S, ld);
	v_psi_0_transform(S, &transformed_a[4 * block_size], block_ld, steps_left - 1, nextWork);

	S6_sum(original_a, S, ld);
	v_psi_0_transform(S, &transformed_a[5 * block_size], block_ld, steps_left - 1, nextWork);

	S7_sum(original_a, S, ld);
	v_psi_0_transform(S, &transformed_a[6 * block_size], block_ld, steps_left - 1, nextWork);

	S8_sum(original_a, S, ld);
	v_psi_0_transform(S, &transformed_a[7 * block_size], block_ld, steps_left - 1, nextWork);

	S9_sum(original_a, S, ld);
	v_psi_0_transform(S, &transformed_a[8 * block_size], block_ld, steps_left - 1, nextWork);

}
