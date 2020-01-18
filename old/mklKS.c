#include <time.h>
#include <stdio.h>
#include <math.h>
#include "mklKS.h"

#define LOOP_COUNT 3
static const int NUM_BLOCKS_REQUIRED = 9;
static const int BASE_CASE_DIM = 2;

int compare_ks_and_mkl(int dim)
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
	double* matrixB = (double*)mkl_malloc(pow * pow * sizeof(double), 64);
	double* matrixC_naive = (double*)mkl_malloc(pow * pow * sizeof(double), 64);

	for (int i = 0; i < n * n; i ++)
	{
		matrixA[i] = 1;
		//matrixA[i + 1] = i;

		matrixB[i] = 1;
		//matrixB[i + 1] = i + 1;
	}
	FILE* fp = fopen("log.txt", "a+");
	printf("Using dim = %d\n", dim);
	fprintf(fp, "Using dim = %d\n", dim);
	clock_t start = clock(), diff;
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		mkl_multiplication(matrixA, matrixB, matrixC_naive, n);
	}
	//printMatrix(matrixC_naive, pow);
	diff = clock() - start;
	double msec = diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	printf("Standard Multiplication took %.2f msecs\n", msec);
	fprintf(fp, "Standard Multiplication took % .2f msecs\n", msec);
	printf("Standard Multiplication took % .2f msecs\n", msec);
	int steps = 8;
	unsigned mem = memoryRequirements(NUM_BLOCKS_REQUIRED, n, BASE_CASE_DIM, steps);
	double* work = (double*)MKL_malloc(mem * sizeof(double), 64);
	start = clock(), diff;
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		ksBasisTransformRec(matrixA, n, steps, work);
		ksBasisTransformRec(matrixB, n, steps, work);
		ksMultRec(matrixA, matrixB, matrixC_naive, n, steps, work);
		ksInverseBasisTransformRec(matrixC_naive, n, steps, work);
	}
	//printMatrix(matrixC_naive, pow);
	diff = clock() - start;
	MKL_free(work);
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	printf("KS (1 step) took %.2f msecs\n", msec);
	fprintf(fp, "KS (1 step) took % .2f msecs\n", msec);

	
	/*mem = memoryRequirements(NUM_BLOCKS_REQUIRED, n, BASE_CASE_DIM, 2);
	work = (double*)MKL_malloc(mem * sizeof(double), 64);
	start = clock(), diff;
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		ksBasisTransformRec(matrixA, n, 2, work);
		ksBasisTransformRec(matrixB, n, 2, work);
		ksMultRec(matrixA, matrixB, matrixC_naive, n, BASE_CASE_DIM, work);
		ksInverseBasisTransformRec(matrixC_naive, n, BASE_CASE_DIM, work);
	}
	//printMatrix(matrixC_naive, pow);
	diff = clock() - start;
	MKL_free(work);
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	printf("KS (2 step) took %.2f msecs\n", msec);
	fprintf(fp, "KS (2 step) took % .2f msecs\n", msec);

	mem = memoryRequirements(NUM_BLOCKS_REQUIRED, n, BASE_CASE_DIM, 3);
	work = (double*)MKL_malloc(mem * sizeof(double), 64);
	start = clock(), diff;
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		ksBasisTransformRec(matrixA, n, 3, work);
		ksBasisTransformRec(matrixB, n, 3, work);
		ksMultRec(matrixA, matrixB, matrixC_naive, n, 3, work);
		ksInverseBasisTransformRec(matrixC_naive, n, 3, work);
	}
	//printMatrix(matrixC_naive, pow);
	diff = clock() - start;
	MKL_free(work);
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	printf("KS (3 step) took %.2f msecs\n", msec);
	fprintf(fp, "KS (3 step) took % .2f msecs\n", msec);*/
	
	// printf("\nStrassen's Multiplication Output:\n");
	//
	// strassensMultRec(matrixA, matrixB, matrixC_strassen, n, 1);
	// printf("Strassen");
	// printMatrix(matrixC_strassen, pow);
	// printf("naive");
	// printMatrix(matrixC_naive, pow);



	mkl_free(matrixA);
	mkl_free(matrixB);
	mkl_free(matrixC_naive);

	return 0;
}

static void S1_sum(double* matrixA, double* S1, int block_size) {
	/*if (block_size <= INTEL_MKL_THRESH) {
		cblas_dcopy(block_size, matrixA, 1, S1, 1);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
		for (int k = 0; k < block_size; k++) {
			S1[k] = matrixA[(3 * block_size) + k];
		}*/
		//memcpy(S1, &matrixA[3 * block_size], sizeof(double) * block_size);
	//}
		S1 = &matrixA[3 * block_size];
}
static void S2_sum(double* matrixA, double* S2, int block_size) {
	/*if (block_size <= INTEL_MKL_THRESH) {
		cblas_dcopy(block_size, matrixA, 1, S2, 1);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
		for (int k = 0; k < block_size; k++) {
			S2[k] = matrixA[(2 * block_size) + k];
		}
	}*/
	//memcpy(S2, &matrixA[2 * block_size], sizeof(double) * block_size);
	S2 = &matrixA[2 * block_size];
}
static void S3_sum(double* matrixA, double* S3, int block_size) {
	/*if (block_size <= INTEL_MKL_THRESH) {
		cblas_dcopy(block_size, matrixA, 1, S3, 1);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
		for (int k = 0; k < block_size; k++) {
			S3[k] = matrixA[(1 * block_size) + k];
		}
	}*/
	//memcpy(S3, &matrixA[1 * block_size], sizeof(double) * block_size);
	S3 = &matrixA[1 * block_size];
}
static void S4_sum(double* matrixA, double* S4, int block_size) {
	/*if (block_size <= INTEL_MKL_THRESH) {
		cblas_dcopy(block_size, matrixA, 1, S4, 1);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
		for (int k = 0; k < block_size; k++) {
			S4[k] = matrixA[(0 * block_size) + k];
		}
	}*/
	S4 = matrixA;
}
static void S5_sum(const double* matrixA, double* S5, int block_size) {
	if (block_size <= INTEL_MKL_THRESH) {
		vdSub(block_size, &matrixA[block_size], &matrixA[2 * block_size], S5);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
		for (int k = 0; k < block_size; k++) {
			S5[k] = matrixA[(1 * block_size) + k] - matrixA[(2 * block_size) + k];
		}
	}
}
static void S6_sum(const double* matrixA, double* S6, int block_size) {
	if (block_size <= INTEL_MKL_THRESH) {
		vdSub(block_size, &matrixA[block_size], matrixA, S6);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
		for (int k = 0; k < block_size; k++) {
			S6[k] = -matrixA[(0 * block_size) + k] + matrixA[(1 * block_size) + k];
		}
	}
}
static void S7_sum(const double* matrixA, double* S7, int block_size) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
		for (int k = 0; k < block_size; k++) {
			S7[k] = -matrixA[(1 * block_size) + k] + matrixA[(3 * block_size) + k];
		}
}
static void T1_sum(double* matrixB, double* T1, int block_size) {
	/*if (block_size <= INTEL_MKL_THRESH) {
		cblas_dcopy(block_size, matrixB, 1, T1, 1);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
		for (int k = 0; k < block_size; k++) {
			T1[k] = matrixB[(3 * block_size) + k];
		}
	}*/
	T1 = &matrixB[3 * block_size];
}
static void T2_sum(double* matrixB, double* T2, int block_size) {
	/*if (block_size <= INTEL_MKL_THRESH) {
		cblas_dcopy(block_size, matrixB, 1, T2, 1);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
		for (int k = 0; k < block_size; k++) {
			T2[k] = matrixB[(2 * block_size) + k];
		}
	}*/
	T2 = &matrixB[2 * block_size];
}
static void T3_sum(double* matrixB, double* T3, int block_size) {
	/*if (block_size <= INTEL_MKL_THRESH) {
		cblas_dcopy(block_size, matrixB, 1, T3, 1);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
		for (int k = 0; k < block_size; k++) {
			T3[k] = matrixB[(1 * block_size) + k];
		}
	}*/
	T3 = &matrixB[1 * block_size];
}
static void T4_sum(double* matrixB, double* T4, int block_size) {
	/*if (block_size <= INTEL_MKL_THRESH) {
		cblas_dcopy(block_size, matrixB, 1, T4, 1);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
		for (int k = 0; k < block_size; k++) {
			T4[k] = matrixB[(0 * block_size) + k];
		}
	}*/
	T4 = matrixB;
}
static void T5_sum(const double* matrixB, double* T5, int block_size) {
	if (block_size <= INTEL_MKL_THRESH) {
		vdSub(block_size, &matrixB[3*block_size], &matrixB[block_size], T5);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
		for (int k = 0; k < block_size; k++) {
			T5[k] = -matrixB[(1 * block_size) + k] + matrixB[(3 * block_size) + k];
		}
	}
}
static void T6_sum(const double* matrixB, double* T6, int block_size) {
	if (block_size <= INTEL_MKL_THRESH) {
		vdSub(block_size, &matrixB[block_size], &matrixB[2*block_size], T6);
}
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
	for (int k = 0; k < block_size; k++) {
		T6[k] = matrixB[(1 * block_size) + k] - matrixB[(2 * block_size) + k];
	}
}
static void T7_sum(const double* matrixB, double* T7, int block_size) {
	if (block_size <= INTEL_MKL_THRESH) {
		vdSub(block_size, &matrixB[block_size], matrixB, T7);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
		for (int k = 0; k < block_size; k++) {
			T7[k] = -matrixB[(0 * block_size) + k] + matrixB[(1 * block_size) + k];
		}
	}
}
static void Q1_sum(double* Q1, int block_size, double* M4, double* M5) {
	if (block_size <= INTEL_MKL_THRESH) {
		vdAdd(block_size, M4, M5, Q1);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
		for (int k = 0; k < block_size; k++) {
			Q1[k] = M4[k] + M5[k];
		}
	}
}
static void Q2_sum(double* Q2, int block_size, double* M3, double* M5, double* M6, double* M7) {
	if (block_size <= INTEL_MKL_THRESH) {
		vdAdd(block_size, M3, M5, Q2);
		vdSub(block_size, Q2, M6, Q2);
		vdAdd(block_size, Q2, M7, Q2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
		for (int k = 0; k < block_size; k++) {
			Q2[k] = M3[k] + M5[k] - M6[k] + M7[k];
		}
	}
}
static void Q3_sum(double* Q3, int block_size, double* M2, double* M7) {
	if (block_size <= INTEL_MKL_THRESH) {
		vdAdd(block_size, M2, M7, Q3);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
		for (int k = 0; k < block_size; k++) {
			Q3[k] = M2[k] + M7[k];
		}
	}
}
static void Q4_sum(double* Q4, int block_size, double* M1, double* M6) {
	if (block_size <= INTEL_MKL_THRESH) {
		vdSub(block_size, M1, M6, Q4);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
		for (int k = 0; k < block_size; k++) {
			Q4[k] = M1[k] - M6[k];
		}
	}
}


/*
* Strassen's Multiplication algorithm using Divide and Conquer technique.
*/
void ksMultRec(double* matrixA, double* matrixB, double* matrixC, int n, int steps_left, double* work)
{
	if (steps_left == 0)
	{
		//This is the terminating condition for using strassen.
		mkl_multiplication(matrixA, matrixB, matrixC, n);
		return;
	}
	clock_t start, diff;
	//Divide the matrix
	double* m1 = work;
	double* m2 = &m1[n / 2 * n / 2];
	double* m3 = &m1[2 * n / 2 * n / 2];
	double* m4 = &m1[3 * n / 2 * n / 2];
	double* m5 = &m1[4 * n / 2 * n / 2];
	double* m6 = &m1[5 * n / 2 * n / 2];
	double* m7 = &m1[6 * n / 2 * n / 2];
	double* S;
	double* T;
	double* nextWork = &work[9 * n / 2 * n / 2];
	int block_size = n / 2 * n / 2;

	// m1 = 1st multiplication
	S1_sum(matrixA, S, block_size);
	T1_sum(matrixB, T, block_size);
	ksMultRec(S, T, m1, n / 2, steps_left - 1, nextWork);

	// m1 = 2nd multiplication
	S2_sum(matrixA, S, block_size);
	T2_sum(matrixB, T, block_size);
	ksMultRec(S, T, m2, n / 2, steps_left - 1, nextWork);

	// m1 = 3rd multiplication
	S3_sum(matrixA, S, block_size);
	T3_sum(matrixB, T, block_size);
	ksMultRec(S, T, m3, n / 2, steps_left - 1, nextWork);

	// m1 = 4th multiplication
	S4_sum(matrixA, S, block_size);
	T4_sum(matrixB, T, block_size);
	ksMultRec(S, T, m4, n / 2, steps_left - 1, nextWork);

	S = &m1[7 * n / 2 * n / 2];
	T = &m1[8 * n / 2 * n / 2];

	double msec;
	// m2 = 5th multiplication
	S5_sum(matrixA, S, block_size);
	T5_sum(matrixB, T, block_size);
	start = clock();
	ksMultRec(S, T, m5, n / 2, steps_left - 1, nextWork);
	diff = clock() - start;
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	//printf("5: %.2f msecs\n", msec);
	

	// m4 = 6th multiplication
	S6_sum(matrixA, S, block_size);
	T6_sum(matrixB, T, block_size);
	start = clock();
	ksMultRec(S, T, m6, n / 2, steps_left - 1, nextWork);
	diff = clock() - start;
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	//printf("6: %.2f msecs\n", msec);
	

	// m3 = 7th multiplication
	start = clock();
	S7_sum(matrixA, S, block_size);
	T7_sum(matrixB, T, block_size);
	start = clock();
	ksMultRec(S, T, m7, n / 2, steps_left - 1, nextWork);
	diff = clock() - start;
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	//printf("7: %.2f msecs\n", msec);

	// creating C11
	start = clock();
	Q1_sum(matrixC, block_size, m4, m5);
	diff = clock() - start;
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	//printf("Q1: %.2f msecs\n", msec);
	// creating C12
	start = clock();
	Q2_sum(&matrixC[block_size], block_size, m3, m5, m6, m7);
	diff = clock() - start;
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	//printf("Q2: %.2f msecs\n", msec);
	start = clock();
	Q3_sum(&matrixC[2*block_size], block_size, m2, m7);
	diff = clock() - start;
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	//printf("Q3: %.2f msecs\n", msec);
	// creating C22
	// done with 1st multiplication, m1 can be reused
	start = clock();
	Q4_sum(&matrixC[3*block_size], block_size, m1, m6);
	diff = clock() - start;
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	//printf("Q4: %.2f msecs\n", msec);

	return;
}
