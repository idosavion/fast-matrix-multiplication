#include <time.h>
#include <stdio.h>
#include <math.h>
#include "ksStrassBasis.h"
#define LOOP_COUNT 3

static const int NUM_BLOCKS_REQUIRED = 2;


int benchmarkKSBasisTransform(int dim)
{
	int i = 0, j = 0, n = 0;
	n = dim;

	//To handle when n is not power of k we do the padding with zero
	int pow = 1;
	while (pow < n)
	{
		pow = pow * 2;
	}
	n = pow;
	double* matrixA = (double*)mkl_malloc(pow * pow * sizeof(double), 64);
	//double* matrixB = (double*)mkl_malloc(pow * pow * sizeof(double), 64);
	double* matrixC_naive = (double*)mkl_malloc(pow * pow * sizeof(double), 64);

	for (int i = 0; i < n * n; i += 2)
	{
		matrixA[i] = 0;
		matrixA[i + 1] = i;

		//matrixB[i] = 0;
		//matrixB[i + 1] = i + 1;
		//matrixC_naive[i] = 0;
	}

	unsigned mem = memoryRequirements(NUM_BLOCKS_REQUIRED, n, 2, 1);
	double* work = (double*)MKL_malloc(mem * sizeof(double), 64);
	clock_t start = clock(), diff;
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		ksBasisTransformRec(matrixA, n, 1, work);
	}
	// printMatrix(matrixC_naive, pow);
	diff = clock() - start;
	MKL_free(work);
	double msec = diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	printf("Basis Transformation (1 step) took %.2f msecs\n", msec);


	mem = memoryRequirements(NUM_BLOCKS_REQUIRED, n, 2, 2);
	work = (double*)MKL_malloc(mem * sizeof(double), 64);
	start = clock(), diff;
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		ksBasisTransformRec(matrixA, n, 2, work);
	}
	// printMatrix(matrixC_naive, pow);
	diff = clock() - start;
	MKL_free(work);
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	printf("Basis Transformation (2 step) took %.2f msecs\n", msec);

	mem = memoryRequirements(NUM_BLOCKS_REQUIRED, n, 2, 3);
	work = (double*)MKL_malloc(mem * sizeof(double), 64);
	start = clock(), diff;
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		ksBasisTransformRec(matrixA, n, 3, work);
	}
	// printMatrix(matrixC_naive, pow);
	diff = clock() - start;
	MKL_free(work);
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	printf("Basis Transformation (3 step) took %.2f msecs\n", msec);

	// printf("\nStrassen's Multiplication Output:\n");
	//
	// strassensMultRec(matrixA, matrixB, matrixC_strassen, n, 1);
	// printf("Strassen");
	// printMatrix(matrixC_strassen, pow);
	// printf("naive");
	// printMatrix(matrixC_naive, pow);



	mkl_free(matrixA);
	//mkl_free(matrixB);
	//mkl_free(matrixC_naive);

	return 0;
}

static void S1_add(const double* matrixA, double* S, int ldA) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatcopy('R', 'N', ldA / 2, ldA / 2, 1, matrixA, ldA, S, ldA / 2);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				S[j * (ldA / 2) + k] = matrixA[j * ldA / 2 + k];
			}
		}
	}
}

static void S2_add(const double* matrixA, const double* tmp, double* S, int ldA) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ldA / 2, ldA / 2, 1, &matrixA[ldA / 2], ldA, 1, tmp, ldA / 2, S, ldA / 2);
	}	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				//S[j * (ldA / 2) + k] = matrixA[ldA * j + (ldA / 2 + k)] - matrixA[ldA * (ldA / 2 + j) + k] + matrixA[(ldA / 2 + j) * ldA + (ldA / 2 + k)];
				S[j * (ldA / 2) + k] = matrixA[ldA * j + (ldA / 2 + k)] + tmp[j*(ldA/2) + k];
			}
		}
	}
}

static void S3_add(const double* matrixA, double* S, int ldA) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ldA / 2, ldA / 2, 1, &matrixA[(ldA / 2) * ldA + (ldA / 2)], ldA, -1, &matrixA[ldA * (ldA / 2)], ldA, S, ldA / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				S[j * (ldA / 2) + k] = matrixA[(ldA / 2 + j) * ldA + (ldA / 2 + k)] - matrixA[ldA * (ldA / 2 + j) + k];
			}
		}
	}
}

static void S4_add(const double* matrixA, double* S, int ldA) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ldA / 2, ldA / 2, 1, &matrixA[ldA / 2], ldA, 1, &matrixA[(ldA / 2) * ldA + (ldA / 2)], ldA, S, ldA / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				S[j * (ldA / 2) + k] = matrixA[ldA * j + (ldA / 2 + k)] + matrixA[(ldA / 2 + j) * ldA + (ldA / 2 + k)];
			}
		}
	}
}

static void T1_add(const double* matrixA, double* S, int ldA) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatcopy('R', 'N', ldA / 2, ldA / 2, 1, matrixA, ldA, S, ldA / 2);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				S[j * (ldA / 2) + k] = matrixA[j * ldA / 2 + k];
			}
		}
	}
}

static void T2_add(const double* matrixA, double* S, int ldA) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ldA / 2, ldA / 2, 1, &matrixA[ldA / 2], ldA, -1, &matrixA[ldA * (ldA / 2)], ldA, S, ldA / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				S[j * (ldA / 2) + k] = matrixA[ldA * j + (ldA / 2 + k)] - matrixA[ldA * (ldA / 2 + j) + k];
			}
		}
	}
}

static void T3_add(const double* matrixA, double* S, int ldA) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ldA / 2, ldA / 2, 1, &matrixA[(ldA / 2) * ldA + (ldA / 2)], ldA, -1, &matrixA[ldA + (ldA / 2)], ldA, S, ldA / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				S[j * (ldA / 2) + k] = matrixA[(ldA / 2 + j) * ldA + (ldA / 2 + k)] - matrixA[ldA * j + (ldA / 2 + k)];
			}
		}
	}
}

static void T4_add(const double* matrixA, const double* tmp, double* S, int ldA) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ldA / 2, ldA / 2, 1, &matrixA[ldA * (ldA / 2)], ldA, 1, tmp, ldA / 2, S, ldA / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				//S[j * (ldA / 2) + k] = matrixA[ldA * (ldA / 2 + j) + k] - matrixA[ldA * j + (ldA / 2 + k)] + matrixA[(ldA / 2 + j) * ldA + (ldA / 2 + k)];
				S[j * (ldA / 2) + k] = matrixA[ldA * (ldA / 2 + j) + k] + tmp[j * (ldA / 2) + k];
			}
		}
	}
}


static void Q1_copy(double* matrixM, double* matrixA, int ldA) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatcopy('R', 'N', ldA / 2, ldA / 2, 1, matrixM, ldA, matrixA, ldA / 2);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				matrixA[j * ldA + k] = matrixM[j * ldA / 2 + k];
			}
		}
	}
}

static void Q2_copy(double* matrixM, double* matrixA, int ldA) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatcopy('R', 'N', ldA / 2, ldA / 2, 1, matrixM, ldA, matrixA, ldA / 2);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				matrixA[j * ldA + (ldA / 2 + k)] = matrixM[j * ldA / 2 + k];
			}
		}
	}
}

static void Q3_copy(double* matrixM, double* matrixA, int ldA) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatcopy('R', 'N', ldA / 2, ldA / 2, 1, matrixM, ldA, matrixA, ldA / 2);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				matrixA[(j + ldA / 2) * ldA + k] = matrixM[j * ldA / 2 + k];
			}
		}
	}
}

static void Q4_copy(double* matrixM, double* matrixA, int ldA) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatcopy('R', 'N', ldA / 2, ldA / 2, 1, matrixM, ldA, matrixA, ldA / 2);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				matrixA[(j + ldA / 2) * ldA + (ldA / 2 + k)] = matrixM[j * ldA / 2 + k];
			}
		}
	}
}

/*
* Strassen's Multiplication algorithm using Divide and Conquer technique.
*/
void ksBasisTransformRec(double* matrixA, int n, int steps_left, double* work)
{
	if (steps_left == 0)
	{
		return;
	}

	//Divide the matrix

	double* S = work;
	double* S2 = &work[n / 2 * n / 2];
	double* nextWork = &work[2 * n / 2 * n / 2];


	//TODO: can use 1 fewer addition
	S1_add(matrixA, S, n);
	ksBasisTransformRec(S, n / 2, steps_left - 1, nextWork);
	// creating C11
	//printMatrix(matrixC, n);
	//MKL_Domatcopy('R', 'N', n / 2, n / 2, 1, m1, n / 2, matrixC, n);
	Q1_copy(S, matrixA, n);

	S3_add(matrixA, S, n);
	S2_add(matrixA, S, S2, n);
	ksBasisTransformRec(S, n / 2, steps_left - 1, nextWork);
	// creating C21
	//MKL_Domatcopy('R', 'N', n / 2, n / 2, 1, m1, n / 2, &matrixC[n * n / 2], n);
	Q3_copy(S, matrixA, n);
	//printMatrix(matrixC, n);

	ksBasisTransformRec(S, n / 2, steps_left - 1, nextWork);
	// creating C12
	Q2_copy(S2, matrixA, n);
	//printMatrix(matrixC, n);

	S4_add(matrixA, S, n);
	ksBasisTransformRec(S, n / 2, steps_left - 1, nextWork);
	// creating C22
	//MKL_Domatcopy('R', 'N', n / 2, n / 2, 1, m1, n / 2, &matrixC[n * n / 2 + n / 2], n);
	Q4_copy(S, matrixA, n);
	//printMatrix(matrixC, n);

}

/*
* Strassen's Multiplication algorithm using Divide and Conquer technique.
*/
void ksInverseBasisTransformRec(double* matrixA, int n, int steps_left, double* work)
{
	if (steps_left == 0)
	{
		return;
	}

	//Divide the matrix

	double* S = work;
	double* S2 = &work[n/2 * n/2];
	double* nextWork = &work[2 * n / 2 * n / 2];


	//TODO: can use 1 fewer addition
	T1_add(matrixA, S, n);
	ksBasisTransformRec(S, n / 2, steps_left - 1, nextWork);
	// creating C11
	//printMatrix(matrixC, n);
	//MKL_Domatcopy('R', 'N', n / 2, n / 2, 1, m1, n / 2, matrixA, n);
	Q1_copy(S, matrixA, n);

	T2_add(matrixA, S, n);
	ksBasisTransformRec(S, n / 2, steps_left - 1, nextWork);
	// creating C12
	Q2_copy(S, matrixA, n);
	//printMatrix(matrixC, n);
	//MKL_Domatcopy('R', 'N', n / 2, n / 2, 1, m1, n / 2, &matrixA[n / 2], n);

	T3_add(matrixA, S, n);
	T4_add(matrixA, S, S2, n);
	ksBasisTransformRec(S, n / 2, steps_left - 1, nextWork);
	// creating C21
	//MKL_Domatcopy('R', 'N', n / 2, n / 2, 1, m1, n / 2, &matrixA[n * n / 2], n);
	Q3_copy(S, matrixA, n);
	//printMatrix(matrixC, n);

	ksBasisTransformRec(S2, n / 2, steps_left - 1, nextWork);
	// creating C22
	//MKL_Domatcopy('R', 'N', n / 2, n / 2, 1, m1, n / 2, &matrixA[n * n / 2 + n / 2], n);
	Q4_copy(S2, matrixA, n);
	//printMatrix(matrixC, n);
}