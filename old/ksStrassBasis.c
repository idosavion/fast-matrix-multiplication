#include <time.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "ksStrassBasis.h"
#define LOOP_COUNT 3

static const int NUM_BLOCKS_REQUIRED = 4;
static const int BASE_CASE_DIM = 2;


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
	//double* matrixC_naive = (double*)mkl_malloc(pow * pow * sizeof(double), 64);
	//int tst = 3;
	for (int i = 0; i < n * n; i ++)
	{
		//if (i == tst) {
			matrixA[i] = 1;
		//}
		//else {
		//	matrixA[i] = 0;
		//}
		//matrixA[i + 1] = i;

		//matrixB[i] = 0;
		//matrixB[i + 1] = i + 1;
		//matrixC_naive[i] = 0;
	}
	//printMatrix(matrixA, pow);
	unsigned mem = memoryRequirements(NUM_BLOCKS_REQUIRED, n, BASE_CASE_DIM, 1);
	printf("%d \n", mem);
	double* work = (double*)MKL_malloc(mem * sizeof(double), 64);
	clock_t start = clock(), diff;
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		ksBasisTransformRec(matrixA, n, 1, work);
		//ksInverseBasisTransformRec(matrixA, n, 1, work);
	}
	//printMatrix(matrixA, pow);
	diff = clock() - start;
	MKL_free(work);
	double msec = diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	printf("Basis Transformation (1 step) took %.2f msecs\n", msec);

	
	mem = memoryRequirements(NUM_BLOCKS_REQUIRED, n, BASE_CASE_DIM, 2);
	printf("%d \n", mem);
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

	mem = memoryRequirements(NUM_BLOCKS_REQUIRED, n, BASE_CASE_DIM, 3);
	printf("%d \n", mem);
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

static void S1_sum(double* matrixA, double* S1, int ld, int stride) {
	/*int block_size = (ld / 2 * ld / 2);
	if ( block_size <= INTEL_MKL_THRESH) {
		cblas_dcopy(block_size, matrixA, 1, S1, 1); // TODO: replace with domatcopy
}
	else {*/
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < (ld / 2); j++) {
			memcpy(&S1[j * (ld / 2)], &matrixA[j * ld], sizeof(double) * stride);
			/*for (int k = 0; k < stride; k++) {
				S1[j * (ld / 2) + k] = matrixA[((ld / 2 * 0) + j) * ld + ((stride * 0) + k)];
			}*/
		}
	//}
}
static void S2_sum(const double* matrixA, double* S2, int ld, int stride) {
	/*int block_size = (ld / 2 * ld / 2);
	if (block_size <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ld / 2, ld / 2, 1, &matrixA[stride], ld, -1, &matrixA[ld / 2 * ld], ld, S2, ld / 2);
		MKL_Domatadd('R', 'N', 'N', ld / 2, ld / 2, 1, S2, ld/2, 1, &matrixA[ld / 2 * ld], ld, S2, ld / 2);
}
	else {*/
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < ld / 2; j++) {
		for (int k = 0; k < stride; k++) {
			S2[j * (ld / 2) + k] = matrixA[((ld / 2 * 0) + j) * ld + ((stride * 1) + k)] - matrixA[((ld / 2 * 1) + j) * ld + ((stride * 0) + k)] + matrixA[((ld / 2 * 1) + j) * ld + ((stride * 1) + k)];
		}
	}
}
static void S3_sum(const double* matrixA, double* S3, int ld, int stride) {
	int block_size = (ld / 2 * ld / 2);
	if (block_size <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ld / 2, ld / 2, 1, &matrixA[ld/2*ld + stride], ld, -1, &matrixA[ld / 2 * ld], ld, S3, ld / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ld / 2; j++) {
			for (int k = 0; k < stride; k++) {
				S3[j * (ld / 2) + k] = -matrixA[((ld / 2 * 1) + j) * ld + ((stride * 0) + k)] + matrixA[((ld / 2 * 1) + j) * ld + ((stride * 1) + k)];
			}
		}
	}
}
static void S4_sum(const double* matrixA, double* S4, int ld, int stride) {
	int block_size = (ld / 2 * ld / 2);
	if (block_size <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ld / 2, ld / 2, 1, &matrixA[stride], ld, 1, &matrixA[ld / 2 * ld + stride], ld, S4, ld / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ld / 2; j++) {
			for (int k = 0; k < stride; k++) {
				S4[j * (ld / 2) + k] = matrixA[((ld / 2 * 0) + j) * ld + ((stride * 1) + k)] + matrixA[((ld / 2 * 1) + j) * ld + ((stride * 1) + k)];
			}
		}
	}
}
static void T1_sum(double* T1, int ld, int stride, double* matrixA1) {
	/*int block_size = (ld / 2 * ld / 2);
	if (block_size <= INTEL_MKL_THRESH) {
		cblas_dcopy(block_size, matrixA1, 1, T1, 1);
}
	else {*/
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ld / 2; j++) {
			memcpy(&T1[j * ld / 2], &matrixA1[j * ld / 2], sizeof(double) * stride);
			/*for (int k = 0; k < stride; k++) {
				T1[(j + ld / 2 * 0) * ld + (stride * 0 + k)] = matrixA1[(ld / 2 * j) + k];
			}*/
		}
	//}
}
static void T2_sum(double* T2, int ld, int stride, double* matrixA2, double* matrixA3) {
	int block_size = (ld / 2 * ld / 2);
	if (block_size <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ld / 2, ld / 2, 1, matrixA2, ld / 2, -1, matrixA3, ld / 2, T2, ld / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ld / 2; j++) {
			for (int k = 0; k < stride; k++) {
				T2[(j + ld / 2 * 0) * ld + (stride * 1 + k)] = matrixA2[(ld / 2 * j) + k] - matrixA3[(ld / 2 * j) + k];
			}
		}
	}
}
static void T3_sum(double* T3, int ld, int stride, double* matrixA2, double* matrixA4) {
	int block_size = (ld / 2 * ld / 2);
	if (block_size <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ld / 2, ld / 2, -1, matrixA2, ld/2, 1, matrixA4, ld/2, T3, ld / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ld / 2; j++) {
			for (int k = 0; k < stride; k++) {
				T3[(j + ld / 2 * 1) * ld + (stride * 0 + k)] = -matrixA2[(ld / 2 * j) + k] + matrixA4[(ld / 2 * j) + k];
			}
		}
	}
}
static void T4_sum(double* T4, int ld, int stride, double* matrixA2, double* matrixA3, double* matrixA4) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ld / 2; j++) {
			for (int k = 0; k < stride; k++) {
				T4[(j + ld / 2 * 1) * ld + (stride * 1 + k)] = -matrixA2[(ld / 2 * j) + k] + matrixA3[(ld / 2 * j) + k] + matrixA4[(ld / 2 * j) + k];
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

static void block_copy(double* M, double* matrixA, int size) {
	/*if (size <= INTEL_MKL_THRESH) {
		cblas_dcopy(size, M, 1, matrixA, 1);
	}
	else {
		/*for (int i = 0; i < size; i++) {
			matrixA[i] = M[i];
		}*/
		memcpy(matrixA, M, sizeof(double) * size);
	//}
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

	double* S1 = work;
	double* S2 = &work[n / 2 * n / 2];
	double* S3 = &work[2*n / 2 * n / 2];
	double* S4 = &work[3*n / 2 * n / 2];
	double* nextWork = &work[4 * n / 2 * n / 2];


	//TODO: can use 1 fewer addition
	S1_sum(matrixA, S1, n, n / 2);
	ksBasisTransformRec(S1, n / 2, steps_left - 1, nextWork);

	S2_sum(matrixA, S2, n, n / 2);
	ksBasisTransformRec(S2, n / 2, steps_left - 1, nextWork);

	S3_sum(matrixA, S3, n, n / 2);
	ksBasisTransformRec(S3, n / 2, steps_left - 1, nextWork);
	// creating C12
	//printMatrix(matrixC, n);
	
	S4_sum(matrixA, S4, n, n / 2);
	ksBasisTransformRec(S4, n / 2, steps_left - 1, nextWork);
	
	//Q1_copy(S1, matrixA, n);
	//Q2_copy(S2, matrixA, n);
	//Q3_copy(S3, matrixA, n);
	//Q4_copy(S4, matrixA, n);
	int s = n / 2 * n / 2;
	block_copy(S1, matrixA, s);
	block_copy(S2, &matrixA[s], s);
	block_copy(S3, &matrixA[2*s], s);
	block_copy(S4, &matrixA[3*s], s);
	/*memcpy(matrixA, S1, sizeof(double) * s);
	memcpy(&matrixA[s], S2, sizeof(double) * s);
	memcpy(&matrixA[2 * s], S3, sizeof(double) * s);
	memcpy(&matrixA[3 * s], S4, sizeof(double) * s);*/

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

	double* S1 = work;
	double* S2 = &work[n / 2 * n / 2];
	double* S3 = &work[2 * n / 2 * n / 2];
	double* S4 = &work[3 * n / 2 * n / 2];
	double* nextWork = &work[4 * n / 2 * n / 2];


	//TODO: can use 1 fewer addition
	int s = n / 2 * n / 2;
	
	
	//T1_sum(matrixA, S1, n, n / 2);
	ksInverseBasisTransformRec(matrixA, n / 2, steps_left - 1, nextWork);
	

	//T2_sum(matrixA, S2, n, n / 2);
	ksInverseBasisTransformRec(&matrixA[s], n / 2, steps_left - 1, nextWork);
	

	//T3_sum(matrixA, S3, n, n / 2);
	ksInverseBasisTransformRec(&matrixA[2*s], n / 2, steps_left - 1, nextWork);

	//T4_sum(matrixA, S4, n, n / 2);
	ksInverseBasisTransformRec(&matrixA[3*s], n / 2, steps_left - 1, nextWork);
	
	block_copy(S1, matrixA, s);
	block_copy(S2, &matrixA[s], s);
	block_copy(S3, &matrixA[2 * s], s);
	block_copy(S4, &matrixA[3 * s], s);
	/*memcpy(matrixA, S1, sizeof(double) * s);
	memcpy(&matrixA[s], S2, sizeof(double) * s);
	memcpy(&matrixA[2*s], S3, sizeof(double) * s);
	memcpy(&matrixA[3 * s], S4, sizeof(double) * s);*/

	T1_sum(matrixA, n, n / 2, S1);
	T2_sum(matrixA, n, n / 2, S2, S3);
	T3_sum(matrixA, n, n / 2, S2, S4);
	T4_sum(matrixA, n, n / 2, S2, S3, S4);
}